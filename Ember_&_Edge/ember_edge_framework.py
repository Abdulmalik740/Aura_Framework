"""
Universal Adaptive Framework — Ember & Edge
Brand identity, special rules, LLM settings — all from brand_config.yaml.
Drop in any brand by swapping the config file.
"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

from brand_consistency_guard import BrandConsistencyGuard
from conversation_state_manager import ConversationStateManager
from context_understanding_engine import ContextUnderstandingEngine
from tone_adaptation_engine import ToneAdaptationEngine
from config_loader import (
    build_brand_ethos, llm as get_llm, special_rules as get_special_rules,
    demo_scenarios as get_demo_scenarios, cfg
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@dataclass
class FrameworkResponse:
    response_text: str
    context_analysis: Dict
    tone_used: Dict
    brand_validation: Dict
    conversation_state: Dict
    confidence: float
    total_time: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    breakdown: Dict = field(default_factory=dict)


class UniversalAdaptiveFramework:
    """
    Brand-agnostic adaptive conversation framework.
    All configuration comes from brand_config.yaml.
    Swap brands by changing the YAML file.
    """

    def __init__(self, client: OpenAI):
        self.client = client
        self._llm = get_llm()
        self._special_rules = get_special_rules()
        self.brand_config = build_brand_ethos()

        brand_boundaries = {
            "core_values": self.brand_config["core_values"],
            "messaging_pillars": self.brand_config["messaging_pillars"],
            "voice_guidelines": self.brand_config["voice_guidelines"],
            "personality": self.brand_config["personality"],
        }

        self.brand_guard = BrandConsistencyGuard(self.brand_config, client)
        self.context_engine = ContextUnderstandingEngine(client)
        self.tone_adapter = ToneAdaptationEngine(brand_boundaries, client)
        self.active_sessions: Dict[str, ConversationStateManager] = {}

    def process_message(self, message: str, session_id: str, user_id: Optional[str] = None) -> FrameworkResponse:
        start_time = time.time()
        breakdown = {}
        total_tokens = 0
        total_cost = 0.0

        # 1. Session
        step_start = time.time()
        conv_state = self.get_or_create_session(session_id, user_id)
        breakdown["1. Session Init"] = {"time": time.time() - step_start, "model": None, "tokens": 0, "cost": 0}

        # 2. Context analysis
        step_start = time.time()
        context = self.context_engine.analyze(
            message=message,
            history=conv_state.turns,
            user_profile=conv_state.cook_profile.__dict__
        )
        step_time = time.time() - step_start
        context_tokens = getattr(context, "tokens_used", 0)
        used_ai = getattr(context, "used_ai", False)
        context_cost = self._calc_cost(self._llm["context_model"], context_tokens) if used_ai else 0
        total_tokens += context_tokens
        total_cost += context_cost
        breakdown["2. Context Analysis"] = {
            "time": step_time,
            "model": self._llm["context_model"] if used_ai else None,
            "tokens": context_tokens,
            "cost": context_cost
        }

        # 3. Special rules (refund / replacement templates)
        step_start = time.time()
        special_response = self._check_special_rules(context, message, conv_state)
        breakdown["3. Special Rules Check"] = {"time": time.time() - step_start, "model": None, "tokens": 0, "cost": 0}

        if special_response:
            conv_state.add_turn(message, special_response, context.to_dict(), {})
            return self._build_response(
                special_response, context, {}, conv_state, 1.0, True,
                time.time() - start_time, total_tokens, total_cost, breakdown
            )

        # 4. Tone selection
        step_start = time.time()
        tone = self.tone_adapter.select_tone(
            context=context.to_dict(),
            conversation_type=context.conversation_type.value,
            risk_flags={
                "churn_risk": conv_state.churn_risk,
                "competitor_switch_risk": conv_state.competitor_switch_risk
            },
        )
        breakdown["4. Tone Selection"] = {"time": time.time() - step_start, "model": None, "tokens": 0, "cost": 0}

        # 5. Generate + validate
        step_start = time.time()
        response_text, brand_validation, gen_tokens, gen_cost = self._generate_validated_response(
            message, context, tone, conv_state
        )
        step_time = time.time() - step_start
        total_tokens += gen_tokens
        total_cost += gen_cost
        breakdown["5. Response Generation + Validation"] = {
            "time": step_time,
            "model": self._llm["generation_model"],
            "tokens": gen_tokens,
            "cost": gen_cost
        }

        # 6. State update
        step_start = time.time()
        conv_state.add_turn(message, response_text, context.to_dict(), tone.to_dict())
        breakdown["6. State Update"] = {"time": time.time() - step_start, "model": None, "tokens": 0, "cost": 0}

        return self._build_response(
            response_text, context, tone, conv_state,
            brand_validation["overall_score"], brand_validation["passed"],
            time.time() - start_time, total_tokens, total_cost, breakdown
        )

    # ── Special rules ────────────────────────────────────────────────────────

    def _check_special_rules(self, context, message: str, conv_state) -> Optional[str]:
        msg_lower = message.lower()
        rules = self._special_rules

        refund = rules["refund"]
        if context.primary_intent.value == refund["intent"]:
            if any(kw in msg_lower for kw in refund["trigger_keywords"]):
                return refund["response_template"]

        replacement = rules["replacement"]
        if context.primary_intent.value == replacement["intent"]:
            product = (
                replacement["default_product_name"]
                if replacement["default_product_keyword"] in msg_lower
                else replacement["default_product_fallback"]
            )
            return replacement["response_template"].format(product=product)

        return None

    # ── Generation + validation loop ─────────────────────────────────────────

    def _generate_validated_response(self, message, context, tone, conv_state, max_attempts=None):
        max_attempts = max_attempts or self._llm["max_validation_attempts"]
        total_tokens = 0
        total_cost = 0.0
        validation_dict = {}
        validation_result = None

        for attempt in range(max_attempts):
            if attempt == 0:
                response_text, gen_tokens = self._generate_response(message, context, tone, conv_state)
            else:
                response_text, gen_tokens = self._regenerate_with_fixes(
                    message, context, tone, validation_result, conv_state
                )
            total_tokens += gen_tokens
            total_cost += self._calc_cost(self._llm["generation_model"], gen_tokens)

            validation_result = self.brand_guard.validate(response_text, context.to_dict())
            val_tokens = getattr(validation_result, "tokens_used", 0)
            validation_dict = {
                "passed": validation_result.passed,
                "overall_score": validation_result.overall_score,
                "violations": getattr(validation_result, "violations", []),
                "tokens_used": val_tokens,
            }
            total_tokens += val_tokens
            if validation_result.passed:
                return response_text, validation_dict, total_tokens, total_cost

        return response_text, validation_dict, total_tokens, total_cost

    def _generate_response(self, message, context, tone, conv_state) -> Tuple[str, int]:
        system_prompt = self._build_system_prompt(message, context, tone, conv_state)

        messages = [{"role": "system", "content": system_prompt}]

        max_history_turns = self._llm.get("context_history_turns", 3)
        if conv_state.turns:
            for turn in conv_state.turns[-max_history_turns:]:
                messages.append({"role": "user", "content": turn.user_message})
                messages.append({"role": "assistant", "content": turn.assistant_response})

        messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self._llm["generation_model"],
            messages=messages,
            temperature=self._llm["generation_temperature"],
            max_tokens=self._llm["generation_max_tokens"],
        )
        tokens = response.usage.total_tokens if hasattr(response, "usage") else 0
        return response.choices[0].message.content.strip(), tokens

    def _build_system_prompt(self, message, context, tone, conv_state) -> str:
        bc = self.brand_config
        tone_instructions = self.tone_adapter.get_generation_instructions(tone, context.to_dict())

        core_values_text = "\n".join(
            f"{i+1}. {v['name'].upper()}: {v['prompt_instruction']}"
            for i, v in enumerate(cfg("core_values"))
        )
        pillars_text = "\n".join(
            f'{i+1}. "{p["text"]}" — {p["prompt_instruction"]}'
            for i, p in enumerate(cfg("messaging_pillars"))
        )

        # Build approved products list from config
        products = cfg("products")
        products_text = "\n".join(
            f"- {name}: best for {', '.join(p['best_for'])} | cook types: {', '.join(p['cook_types'])}"
            for name, p in products.items()
        )

        # Build recommendation guide from rules
        rules = cfg("recommendation_rules") or {}
        rule_map = {
            "fear_safety":       "scared/nervous/afraid of knives → Paring Knife",
            "professional_chef": "professional chef/volume prep → Artisan Chef's Knife",
            "bread_pastry":      "bread/pastry/baking → Bread Knife",
            "vegetable_focus":   "vegetables/fish/thin slicing → Santoku",
            "beginner":          "beginner/just started/first knife → Beginner Set",
            "default":           "general/unclear → Artisan Chef's Knife",
        }
        rec_lines = [f"  • {label}" for key, label in rule_map.items() if key in rules]
        recommendation_guide = "\n".join(rec_lines)

        # Detect complaint / fear context to suppress poetic language
        intent = context.primary_intent.value
        emotion = context.primary_emotion
        is_complaint = intent in ("complaint", "return_refund", "replacement", "product_issue")
        is_fear = intent == "fear_concern" or emotion == "fearful"
        is_purchase = intent == "purchase_intent"

        if is_complaint:
            mode_instruction = """🚨 COMPLAINT MODE — STRICT RULES:
- NO sensory language, NO poetry, NO metaphors
- Acknowledge the problem directly and briefly
- State the concrete solution immediately
- Be calm and decisive, NOT corporate
- EXAMPLE: "That shouldn't happen. Bring your receipt to any store — we'll sort it on the spot."
- NEVER say "we apologize for the inconvenience" or "I'm so sorry to hear that"
"""
        elif is_fear:
            mode_instruction = """⚠️ FEAR / SAFETY MODE — STRICT RULES:
- Calm, reassuring tone — patient instructor, not dismissive
- Recommend the Paring Knife — smaller blade builds confidence
- DO NOT ask "what do you feel?" — user hasn't held the knife yet
- Gentle, practical guidance only. No poetic immersion.
- EXAMPLE: "Start with the Paring Knife. Smaller blade, full control. Confidence comes before the chef's knife."
"""
        elif is_purchase:
            mode_instruction = """💳 PURCHASE MODE — STRICT RULES:
- 60-80 words. Name the knife. State key facts. One gentle close.
- NO "amazing deal", "don't miss out", "limited stock" — no urgency pressure
- Be decisive and confident, not salesy
"""
        else:
            mode_instruction = """✨ BRAND VOICE MODE:
- Sensory, flowing sentences (8-15 words average)
- Lead with feel, weight, balance — technique follows sensation
- Redirect to the ingredient before the knife
- Invite reflection with one gentle question at the close
- Be decisive — name the specific knife when recommending
"""

        return f"""You are a {bc['name']} customer guide — a patient chef-instructor with 20 years in Michelin kitchens.

{mode_instruction}

=== BRAND ETHOS ===

PERSONALITY: {bc['personality']}
TAGLINE: "{bc['tagline']}" — never say this in responses.
MISSION: {bc['mission']}

CORE VALUES (apply all):
{core_values_text}

MESSAGING PILLARS (weave naturally):
{pillars_text}

VOICE DO:
{chr(10).join('  ✓ ' + rule for rule in bc['voice_guidelines']['do'])}

VOICE DON'T:
{chr(10).join('  ✗ ' + rule for rule in bc['voice_guidelines']['dont'])}

=== TONE & CONTEXT ===

{tone_instructions}

Cook type: {context.cook_type}
Emotion: {context.primary_emotion}
Intent: {intent}
Cooking context: {context.training_context or 'general'}

=== APPROVED PRODUCTS ONLY ===
You may ONLY recommend these exact products by these exact names. NEVER invent or mention any other knife:
{products_text}

RECOMMENDATION GUIDE — follow this exactly:
{recommendation_guide}
If no rule matches, recommend the closest product from the approved list above.

=== CRITICAL REQUIREMENTS ===

- WORD COUNT: 60 words. Count them. Not fewer, not more.
- SENTENCES: 8-15 words average. Natural and flowing, not choppy.
- NEVER use: "Does that make sense?", "Does that help?", "Let me know if", "Feel free to"
- NEVER use technical specs: HRC, Rockwell, blade angles, geometry
- NEVER use superlatives: "best-in-class", "revolutionary", "amazing"
- ALWAYS name a specific product when making a recommendation — from the approved list ONLY
- NEVER mention competitor brands

GOOD EXAMPLE (product question): "Before the knife, tell me — what do you find yourself reaching for most in the kitchen? A ripe tomato has a different story than a winter squash. Start there. I'd begin with the Artisan Chef's Knife; it handles most of what a kitchen asks, and you'll feel the difference in the first cut."
BAD EXAMPLE: "Thank you for your question! Here are our top knives: 1. Chef's knife 2. Santoku 3. Paring knife. HRC 60, 15-degree angle..." (corporate, spec-driven, numbered list)

=== YOUR RESPONSE (60 WORDS) ==="""

    # ── Regeneration with fixes ───────────────────────────────────────────────

    def _regenerate_with_fixes(self, message, context, tone, validation_result, conv_state) -> Tuple[str, int]:
        if isinstance(validation_result, dict):
            violations = validation_result.get("violations", [])
        else:
            violations = getattr(validation_result, "violations", [])

        issues = []
        for v in violations[:3]:
            if isinstance(v, dict):
                issues.append(f"❌ {v.get('description', str(v))}")
                if v.get("suggestion"):
                    issues.append(f"✅ Fix: {v['suggestion']}")
            else:
                issues.append(f"❌ {getattr(v, 'description', str(v))}")
                if hasattr(v, "suggestion") and v.suggestion:
                    issues.append(f"✅ Fix: {v.suggestion}")

        bc = self.brand_config
        core_values = ", ".join(v["name"] for v in cfg("core_values"))

        system_prompt = f"""You are an {bc['name']} customer guide — patient chef-instructor voice.

⚠️ PREVIOUS RESPONSE VIOLATED BRAND — FIX THESE ISSUES:

{chr(10).join(issues) if issues else 'General brand violation — too corporate or too technical.'}

CORE VALUES TO APPLY: {core_values}

RULES:
- 60 words exactly
- Flowing sentences, 8-15 words average
- NO technical specs (HRC, angles, geometry)
- NO corporate phrases ("Does that make sense?", "Feel free to", "Let me know")
- NO poetry or metaphors if user is complaining
- Name a specific knife if recommending
- Sensory language where appropriate: feel, weight, balance, glide, whisper

USER MESSAGE: "{message}"
COOK TYPE: {context.cook_type}
EMOTION: {context.primary_emotion}

Generate corrected {bc['name']} response. 70 words."""

        messages = [{"role": "system", "content": system_prompt}]

        max_history_turns = self._llm.get("context_history_turns", 3)
        if conv_state and conv_state.turns:
            for turn in conv_state.turns[-max_history_turns:]:
                messages.append({"role": "user", "content": turn.user_message})
                messages.append({"role": "assistant", "content": turn.assistant_response})

        messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self._llm["regeneration_model"],
            messages=messages,
            temperature=self._llm["regeneration_temperature"],
            max_tokens=self._llm["regeneration_max_tokens"],
        )
        tokens = response.usage.total_tokens if hasattr(response, "usage") else 0
        return response.choices[0].message.content.strip(), tokens

    # ── Utilities ────────────────────────────────────────────────────────────

    def _calc_cost(self, model: str, tokens: int) -> float:
        if not model or tokens == 0:
            return 0.0
        cost_map = self._llm.get("cost_per_1k_tokens", {})
        for key, price in cost_map.items():
            if key in model.lower():
                return (tokens / 1000) * price
        return 0.0

    def get_or_create_session(self, session_id: str, user_id: Optional[str] = None) -> ConversationStateManager:
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = ConversationStateManager(session_id, user_id)
        return self.active_sessions[session_id]

    def _build_response(
        self, response_text, context, tone, conv_state,
        confidence, brand_passed, total_time, total_tokens, total_cost, breakdown
    ) -> FrameworkResponse:
        tone_dict = tone.to_dict() if hasattr(tone, "to_dict") else {}
        return FrameworkResponse(
            response_text=response_text,
            context_analysis=context.to_dict(),
            tone_used=tone_dict,
            brand_validation={"passed": brand_passed, "overall_score": confidence},
            conversation_state=conv_state.get_context_summary(),
            confidence=confidence,
            total_time=total_time,
            total_tokens=total_tokens,
            total_cost=total_cost,
            breakdown=breakdown,
        )


# ── Backwards-compatible alias ────────────────────────────────────────────────
EmberEdgeFramework = UniversalAdaptiveFramework


# ── Demo ──────────────────────────────────────────────────────────────────────
def run_demo():
    brand = build_brand_ethos()
    scenarios = get_demo_scenarios()

    print("\n" + "=" * 80)
    print(f"🔪 {brand['name'].upper()} — UNIVERSAL ADAPTIVE FRAMEWORK")
    print("=" * 80)
    print(f"Mission: {brand['mission']}")
    print(f"Values: {', '.join(brand['core_values'])}")
    print("=" * 80)

    framework = UniversalAdaptiveFramework(client)

    for scenario in scenarios:
        print(f"\n{'='*80}\nSESSION: {scenario['session_id']}\n{'='*80}\n")
        message = scenario["message"]
        print(f"Cook: {message}\n")

        response = framework.process_message(message=message, session_id=scenario["session_id"])

        print(f"{brand['name']}: {response.response_text}")
        print(f"\nValidation:")
        print(f"  Brand: {'✅' if response.brand_validation['passed'] else '❌'} {response.brand_validation['overall_score']:.0%}")
        print(f"  Emotion: {response.context_analysis.get('emotion')}")
        print(f"  Intent:  {response.context_analysis.get('intent')}")
        print(f"\nPerformance:")
        print(f"  Time: {response.total_time:.2f}s  |  Tokens: {response.total_tokens}  |  Cost: ${response.total_cost:.6f}")
        print(f"\n  Step Breakdown:")
        for step_name, metrics in response.breakdown.items():
            model_str = metrics["model"] if metrics["model"] else "No API call"
            print(f"    {step_name:45s} | {metrics['time']:.3f}s | {model_str}")
        print()


if __name__ == "__main__":
    run_demo()