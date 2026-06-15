"""
Universal Adaptive Framework
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
        context = self.context_engine.analyze(message=message, history=conv_state.turns, user_profile=conv_state.athlete_profile.__dict__)
        step_time = time.time() - step_start
        context_tokens = getattr(context, "tokens_used", 0)
        used_ai = getattr(context, "used_ai", False)
        context_cost = self._calc_cost(self._llm["context_model"], context_tokens) if used_ai else 0
        total_tokens += context_tokens
        total_cost += context_cost
        breakdown["2. Context Analysis"] = {"time": step_time, "model": self._llm["context_model"] if used_ai else None, "tokens": context_tokens, "cost": context_cost}

        # 3. Special rules
        step_start = time.time()
        special_response = self._check_special_rules(context, message, conv_state)
        breakdown["3. Special Rules Check"] = {"time": time.time() - step_start, "model": None, "tokens": 0, "cost": 0}

        if special_response:
            conv_state.add_turn(message, special_response, context.to_dict(), {})
            return self._build_response(special_response, context, {}, conv_state, 1.0, True,
                                        time.time() - start_time, total_tokens, total_cost, breakdown)

        # 4. Tone selection
        step_start = time.time()
        tone = self.tone_adapter.select_tone(
            context=context.to_dict(),
            conversation_type=context.conversation_type.value,
            risk_flags={"churn_risk": conv_state.churn_risk, "competitor_switch_risk": conv_state.competitor_switch_risk},
        )
        breakdown["4. Tone Selection"] = {"time": time.time() - step_start, "model": None, "tokens": 0, "cost": 0}

        # 5. Generate + validate
        step_start = time.time()
        response_text, brand_validation, gen_tokens, gen_cost = self._generate_validated_response(message, context, tone, conv_state)
        step_time = time.time() - step_start
        total_tokens += gen_tokens
        total_cost += gen_cost
        breakdown["5. Response Generation + Validation"] = {"time": step_time, "model": f"{self._llm['generation_model']}", "tokens": gen_tokens, "cost": gen_cost}

        # 6. State update
        step_start = time.time()
        conv_state.add_turn(message, response_text, context.to_dict(), tone.to_dict())
        breakdown["6. State Update"] = {"time": time.time() - step_start, "model": None, "tokens": 0, "cost": 0}

        return self._build_response(response_text, context, tone, conv_state,
                                    brand_validation["overall_score"], brand_validation["passed"],
                                    time.time() - start_time, total_tokens, total_cost, breakdown)

    def _check_special_rules(self, context, message: str, conv_state) -> Optional[str]:
        """
        Handles three-step product issue flow:
          1. Detect product issue  → ask "refund or replacement?"
          2. User replies          → confirm + fire callback
          3. Direct refund/replacement intents → handle immediately
        """
        msg_lower = message.lower()
        rules = self._special_rules

        # ── Step 2: We are waiting for refund/replacement answer ─────────────
        pending = getattr(conv_state, "_pending_clarification", None)
        if pending == "awaiting_refund_or_replacement":
            refund_rule = rules["refund"]
            replacement_rule = rules["replacement"]

            wants_refund = any(kw in msg_lower for kw in refund_rule["clarification_trigger_keywords"])
            wants_replacement = any(kw in msg_lower for kw in replacement_rule["clarification_trigger_keywords"])

            if wants_refund:
                conv_state._pending_clarification = None
                self._fire_callback(
                    refund_rule["callback"],
                    conv_state,
                    refund_rule["callback_payload_fields"],
                )
                return refund_rule["response_template"]

            if wants_replacement:
                conv_state._pending_clarification = None
                product = (
                    replacement_rule["default_product_name"]
                    if replacement_rule["default_product_keyword"] in msg_lower
                    else replacement_rule["default_product_fallback"]
                )
                self._fire_callback(
                    replacement_rule["callback"],
                    conv_state,
                    replacement_rule["callback_payload_fields"],
                )
                return replacement_rule["response_template"].format(product=product)

            # Unclear answer — ask again
            return "Just to confirm — do you want a refund or a replacement pair?"

        # ── Step 1: Detect product issue → ask clarification ─────────────────
        product_issue = rules.get("product_issue")
        if product_issue and context.primary_intent.value == product_issue["intent"]:
            if any(kw in msg_lower for kw in product_issue["trigger_keywords"]):
                conv_state._pending_clarification = product_issue["awaiting_clarification_state"]
                return product_issue["clarification_message"]

        # ── Direct refund intent — only fire template when complaint context exists ──
        # Pure process questions ("how do I get a refund?") fall through to LLM.
        # Escalation signals (legal/sue) always fall through to LLM so the injury
        # acknowledgment and escalation behaviour rules can fire properly.
        # Template fires only when: complaint context present AND no escalation signals.
        refund = rules["refund"]
        if context.primary_intent.value == refund["intent"]:
            if any(kw in msg_lower for kw in refund["trigger_keywords"]):
                has_escalation = bool(context.escalation_indicators)
                has_complaint_context = (
                    conv_state.escalation_risk > 0
                    or conv_state.frustration_counter > 0
                    or any(t.intent in ("product_issue", "complaint") for t in conv_state.turns)
                    or context.primary_emotion in ("angry", "frustrated")
                )
                if has_complaint_context and not has_escalation:
                    self._fire_callback(refund["callback"], conv_state, refund["callback_payload_fields"])
                    return refund["response_template"]
                # Escalation present or no complaint context — fall through to LLM

        # ── Direct replacement intent — only fire template when complaint context exists ──
        # Pure process questions ("can I get a replacement?") fall through to LLM.
        # Escalation signals (legal/sue) always fall through to LLM so the injury
        # acknowledgment and escalation behaviour rules can fire properly.
        # Template fires only when: complaint context present AND no escalation signals.
        replacement = rules["replacement"]
        if context.primary_intent.value == replacement["intent"]:
            has_escalation = bool(context.escalation_indicators)
            has_complaint_context = (
                conv_state.escalation_risk > 0
                or conv_state.frustration_counter > 0
                or any(t.intent in ("product_issue", "complaint") for t in conv_state.turns)
                or context.primary_emotion in ("angry", "frustrated")
            )
            product = (
                replacement["default_product_name"]
                if replacement["default_product_keyword"] in msg_lower
                else replacement["default_product_fallback"]
            )
            if has_complaint_context and not has_escalation:
                self._fire_callback(replacement["callback"], conv_state, replacement["callback_payload_fields"])
                return replacement["response_template"].format(product=product)
            # Escalation present or no complaint context — fall through to LLM

        return None

    def _fire_callback(self, callback_name: str, conv_state, payload_fields: list):
        """
        Dispatch a named callback with session context.

        Override this method to connect your real backend systems
        (order management, refund API, warehouse, etc.).

        The callback_name comes from brand_config.yaml → special_rules → callbacks.
        Add your actual integrations here:

            if callback_name == "process_refund":
                refund_api.create(order_id=..., session_id=conv_state.session_id)
            elif callback_name == "process_replacement":
                warehouse_api.ship_replacement(session_id=conv_state.session_id)
        """
        payload = {}
        if "session_id" in payload_fields:
            payload["session_id"] = conv_state.session_id
        if "user_id" in payload_fields:
            payload["user_id"] = conv_state.user_id
        if "product_mentioned" in payload_fields:
            payload["product_mentioned"] = (
                conv_state.mentioned_products[-1] if conv_state.mentioned_products else None
            )
        # Log the callback — wire your backend here
        print(f"\U0001f514 CALLBACK FIRED: {callback_name} | payload={payload}")

    def _generate_validated_response(self, message, context, tone, conv_state, max_attempts=None):
        max_attempts = max_attempts or self._llm["max_validation_attempts"]
        total_tokens = 0
        total_cost = 0.0
        validation_dict = {}

        for attempt in range(max_attempts):
            if attempt == 0:
                response_text, gen_tokens = self._generate_response(message, context, tone, conv_state)
            else:
                response_text, gen_tokens = self._regenerate_with_fixes(message, context, tone, validation_result, conv_state)
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
        
        # Build messages array with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Include recent conversation turns for context (last 3 turns)
        max_history_turns = self._llm.get("context_history_turns", 3)
        if conv_state.turns:
            recent_turns = conv_state.turns[-max_history_turns:]
            for turn in recent_turns:
                messages.append({"role": "user", "content": turn.user_message})
                messages.append({"role": "assistant", "content": turn.assistant_response})
        
        # Add current message
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
            f"{i+1}. {v['name'].upper()}: {v['name']} - {v['prompt_instruction']}"
            for i, v in enumerate(cfg("core_values"))
        )
        pillars_text = "\n".join(
            f'{i+1}. "{p["text"]}" - {p["prompt_instruction"]}'
            for i, p in enumerate(cfg("messaging_pillars"))
        )

        # Build approved products list from config
        products = cfg("products")
        products_text = "\n".join(
            f"- {name}: best for {', '.join(p['best_for'])} | athlete types: {', '.join(p['athlete_types'])}"
            for name, p in products.items()
        )

        # Build recommendation guide from rules
        rules = cfg("recommendation_rules") or {}
        rec_lines = []
        rule_map = {
            "injury":               "injury/pain/shin splints/recovering → Recovery Stride",
            "marathon_non_beginner":"marathon/26.2/half marathon (non-beginner) → Marathon Pro",
            "marathon_beginner":    "marathon (beginner) → Runner 5",
            "speed_work":           "speed/race/5K competitive/10K competitive → Velocity X2",
            "competitive_10k":      "10K training → Velocity X2",
            "ultra_trail":          "trail/ultra → Ultra Trail",
            "comeback":             "comeback/injury recovery → Recovery Stride",
            "beginner_training":    "beginner/just started → Runner 5",
            "athlete_elite":        "elite/competitive athlete → Velocity X2",
            "athlete_5am_club":     "5am club/consistent daily → Runner 5",
            "athlete_comeback":     "comeback athlete → Recovery Stride",
            "athlete_beginner_casual": "beginner/casual → Runner 5",
            "default":              "unclear/general → Runner 5",
        }
        for key, label in rule_map.items():
            if key in rules:
                rec_lines.append(f"  • {label}")
        recommendation_guide = "\n".join(rec_lines)

        return f"""You are a {bc['name']} support agent.

    === COMPLETE BRAND ETHOS ===

    BRAND DNA: {bc['personality']}
    Tagline: {bc['tagline']}
    Mission: {bc['mission']}

    CORE VALUES (Apply ALL):
    {core_values_text}

    MESSAGING PILLARS (Weave naturally):
    {pillars_text}

    VOICE GUIDELINES:

    DO:
    {chr(10).join('  ✓ ' + rule for rule in bc['voice_guidelines']['do'])}

    DON'T:
    {chr(10).join('  ✗ ' + rule for rule in bc['voice_guidelines']['dont'])}

    === APPROVED PRODUCTS ONLY ===
    You may ONLY recommend these exact products by these exact names. NEVER invent or mention any other product:
    {products_text}

    RECOMMENDATION GUIDE — follow this exactly:
{recommendation_guide}
    If no rule matches, recommend the closest product from the approved list above.

    CRITICAL REQUIREMENTS:
    - CONTRACTIONS: you're, don't, can't, it's, we're
    - ACTION VERBS: Go, Push, Start, Move, Train, Earn
    - NO explanations, NO mission statements, NO brand speeches
    - NO "we're about...", NO "our focus is...", NO summaries of brand values
    - when asked general question do not recommend any shoe.
    - NEVER mention competitor brands
    - ALWAYS name a specific product if recommending gear — from the approved list above ONLY
    - "{bc['tagline']}" is our tagline — never say it in responses
    - NEVER use: "starting strong", "claiming your greatness", "crush it", "you've got this", "you're amazing", "on your way"

    === WORD LIMIT BY INTENT ===
    complaint / product_issue / return_refund / replacement → MAX 35 words (own it fast, resolve it fast)
    motivation_seeking / training_question                  → MAX 40 words (challenge + one concrete action)
    product_question / sizing_help / purchase_intent        → MAX 30 words (name product, two feature, one action)
    competition_comparison → MAX 30 words. Lead with "You vs. You" philosophy, then ONE concrete trust signal (e.g. athlete community, product quality, warranty). Never name competitors. Never compare features.
    praise / closing                                        → MAX 20 words (brief, warm, move on)

    === RESPONSE STRUCTURE BY INTENT ===
    complaint     → ownership first, resolution immediately. Fragment style.
    motivation    → one concrete physical action for today. No platitudes.
    product_question → exact product name first, one reason, one imperative.
    injury        → acknowledge setback, name the product, one comeback action.
    return_refund → steps in brand voice, no corporate language.

    competition_comparison — STRUCTURE: fragments only, under 20 words total.
      CRITICAL: even if the athlete is ANGRY, this is NOT a complaint. Do NOT use
      "That's on us." Do NOT apologise. The intent is comparison, not service issue.
      Angry emotion + competition_comparison = hold philosophy calmly, don't own anything.

      FOUR fragments in order:
      [1] Hook: 1 line tied to their specific situation (5 words max)
      [2] You vs. You redirect (4 words max): "You vs. you." / "Only opponent: you."
      [3] Product OR trust fragment — pick based on their question type:
            Price/value question → name specific product + 1 word reason
            Philosophy/rejection → 3-word trust fragment ("Built to last." / "Backed always.")
      [4] Imperative (2-4 words): "Prove it." / "Lace up." / "Move."

      SPECIAL CASES:
      Athlete says they PREFER another brand / brand isn't for them:
        DO NOT argue. DO NOT try to win them back. Respect it. Hold philosophy.
        "Not every athlete is built for this. You vs. you. Move."
      Athlete is ANGRY + comparing:
        Do not own anything. Do not apologise. Hold the philosophy calmly.
        "You vs. you. That's the only scoreboard. Runner 5. Earn it."
      Beginner asking about another brand:
        Welcome warmly. Never confirm or deny the competitor's claim.
        "Everyone starts somewhere. Runner 5. Built for day one. Lace up."

      Examples (price-driven):
        "You vs. you. Runner 5. Built for the grind. Prove it."
        "The only competition is you. Runner 5 — show up daily."
      Examples (rejection):
        "Not every athlete is built for this. You vs. you. Move."
        "Respect. You vs. you — that's all that matters. Keep running."

      Never reuse the same opening. Fragment style always. No full sentences.

    === TONE & CONTEXT ===

    {tone_instructions}

    Athlete: {context.athlete_type}, {context.primary_emotion}, {context.motivation_level} motivation, relationship: {conv_state.athlete_profile.relationship_stage}
    Intent: {context.primary_intent.value}
    Training: {context.training_context or 'general'}
    Injury mentioned: {context.injury_mentioned}
    Escalation signals: {", ".join(context.escalation_indicators) if context.escalation_indicators else "none"}
    Escalation risk: {"HIGH" if conv_state.escalation_risk >= 0.7 else "MEDIUM" if conv_state.escalation_risk >= 0.3 else "none"}
    Churn risk: {"HIGH" if conv_state.churn_risk >= 0.5 else "none"}

    === INTENT → BEHAVIOUR (follow this exactly) ===
    If escalation risk is HIGH or escalation signals present (legal/sue/lawyer/bbb):
      → If injury_mentioned: ONE line only — name the specific failure and own it in
        the same sentence. This line IS the ownership. Do NOT add "That's on us" after.
        e.g. "That's a real failure on our part." or "Six months of training and a shoe
        failure — that's on us." One sentence. Then resolution path immediately.
      → If no injury: own it in one line ("That's on us."). State resolution. Stop.
      → Resolution path — match what they explicitly asked for:
          If intent is return_refund: "Returns section on our website — order number
            and a photo. Refund processed in 3-5 days."
          If intent is replacement: "Order number and a photo — we'll ship a
            replacement. Tracking in 24 hours."
          If intent unclear: "Order number and a photo — refund or replacement,
            your call."
      → Do NOT acknowledge the legal threat. Do NOT promise timelines.
      → Do NOT recommend products. Do NOT apologise beyond ownership.
      → Silence on the legal threat is the correct response. Move to fix only.
    If intent is complaint/product_issue: open with ownership ("That's on us."), state resolution immediately, no softening preamble.
    If intent is return_refund AND complaint context (anger/prior product issue): own it first, then give process. "That's on us. Returns — order number + photo."
    If intent is return_refund AND no complaint context (process question only): give steps in brand voice, no ownership statement. "Returns section on our website. Order number ready. Ship back — refund in 3–5 days."
    If intent is motivation_seeking or motivation is low: no platitudes, give ONE specific physical action the athlete can do today.
    If intent is product_question/sizing_help: name the exact product first, then two features, then one imperative. Nothing else.
    If intent is training_question:
      Give 2-3 punchy training principles in brand voice. Redirect to the relevant
      shoe for their context. Do NOT write a full plan — out of scope. Max 40 words.
      End with one imperative.
    If intent is general_chat AND topic is sensitive (mental health, burnout, anxiety
      about running, grief, identity): support tone. Engage genuinely in 1-2 short
      lines. No platitudes. No deflection ("we're a footwear brand"). Brand voice.
    If intent is competition_comparison: [1] hook tied to athlete's specific situation, [2] internal-competition redirect (no brand names), [3] one trust signal relevant to their context, [4] short imperative. Never compare features. Never name competitors. Fresh phrasing every time.
    If intent is purchase_intent: state availability + one fact. Done.
    If intent is general_chat AND question is factual/logistical (shipping, stores,
      availability, locations, policies, prices not in config):
      NEVER invent an answer. NEVER state a fact you don't have.
      Redirect to website in brand voice. Max 10 words. No apology.
      "Check our website — fastest answer." or "Website has the details. Go."
      Do NOT say "we don't ship to X" or "we do ship to X" — you don't know.
    If injury_mentioned AND no escalation: recommend Recovery Stride, one concrete comeback action.

    === YOUR RESPONSE ===
    Follow the word limit for this intent. Punchy. Direct. No speeches."""

    def _regenerate_with_fixes(self, message, context, tone, validation_result, conv_state) -> Tuple[str, int]:
        if isinstance(validation_result, dict):
            violations = validation_result.get("violations", [])
        else:
            violations = getattr(validation_result, "violations", [])

        issues = []
        for v in violations[:3]:
            if isinstance(v, dict):
                issues.append(f" {v.get('description', str(v))}")
                if v.get("suggestion"):
                    issues.append(f"✅ {v['suggestion']}")
            else:
                issues.append(f" {getattr(v, 'description', str(v))}")
                if hasattr(v, "suggestion") and v.suggestion:
                    issues.append(f"✅ {v.suggestion}")

        intent_limits = {
            "complaint": 35, "product_issue": 35, "return_refund": 35, "replacement": 35,
            "motivation_seeking": 40, "training_question": 40,
            "product_question": 30, "sizing_help": 30, "purchase_intent": 30,
            "competition_comparison": 35, "praise": 20, "closing": 20,
        }
        word_limit = intent_limits.get(context.primary_intent.value, 40)

        bc = self.brand_config
        core_values = ", ".join(v["name"] for v in cfg("core_values"))
        system_prompt = f"""You are a {bc['name']} support agent.

 PREVIOUS RESPONSE VIOLATED BRAND ETHOS:

{chr(10).join(issues) if issues else 'General brand violation - be more direct and punchy'}

CRITICAL: Follow ALL core values: {core_values}
WORD LIMIT: MAX {word_limit} words for intent '{context.primary_intent.value}'.
SHORT sentences (4-7 words). CONTRACTIONS (you're, don't). ACTION verbs.

USER: "{message}"
ATHLETE: {context.primary_emotion}, {context.motivation_level} motivation

Generate corrected {bc['name']} response."""

        # Build messages with history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Include recent conversation for context
        max_history_turns = self._llm.get("context_history_turns", 3)
        if conv_state and conv_state.turns:
            recent_turns = conv_state.turns[-max_history_turns:]
            for turn in recent_turns:
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

    def _calc_cost(self, model: str, tokens: int) -> float:
        if not model or tokens == 0:
            return 0.0
        pricing = self._llm.get("pricing", {})
        model_pricing = pricing.get(model, pricing.get("gpt-4o-mini", {"prompt": 0.075, "completion": 0.3}))
        blended = (model_pricing["prompt"] + model_pricing["completion"]) / 2
        return (tokens / 1_000_000) * blended

    def get_or_create_session(self, session_id: str, user_id: Optional[str] = None) -> ConversationStateManager:
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = ConversationStateManager(session_id, user_id)
        return self.active_sessions[session_id]

    def _build_response(self, response_text, context, tone, conv_state, confidence, brand_passed, total_time, total_tokens, total_cost, breakdown) -> FrameworkResponse:
        tone_dict = tone.to_dict() if hasattr(tone, "to_dict") else {}
        return FrameworkResponse(
            response_text=response_text, context_analysis=context.to_dict(), tone_used=tone_dict,
            brand_validation={"passed": brand_passed, "overall_score": confidence},
            conversation_state=conv_state.get_context_summary(), confidence=confidence,
            total_time=total_time, total_tokens=total_tokens, total_cost=total_cost, breakdown=breakdown,
        )


# ── Backwards-compatible alias ────────────────────────────────────────────────
ApexStrideFramework = UniversalAdaptiveFramework


# ── Demo ──────────────────────────────────────────────────────────────────────
def run_demo():
    brand = build_brand_ethos()
    scenarios = get_demo_scenarios()

    print("\n" + "=" * 80)
    print(f"🚀 {brand['name'].upper()} - UNIVERSAL ADAPTIVE FRAMEWORK")
    print("=" * 80)
    print(f"Mission: {brand['mission']}")
    print(f"Values: {', '.join(brand['core_values'])}")
    print("=" * 80)

    framework = UniversalAdaptiveFramework(client)

    for scenario in scenarios:
        print(f"\n{'='*80}\nSESSION: {scenario['session_id']}\n{'='*80}\n")
        message = scenario["message"]
        print(f"Athlete: {message}")

        response = framework.process_message(message=message, session_id=scenario["session_id"])

        print(f"{brand['name']}: {response.response_text}")
        print(f"\nValidation:")
        print(f"  Brand: {'✅' if response.brand_validation['passed'] else '❌'} {response.brand_validation['overall_score']:.0%}")
        print(f"  Emotion: {response.context_analysis.get('emotion')}")
        print(f"  Intent: {response.context_analysis.get('intent')}")
        print(f"\nPerformance:")
        print(f"  Time: {response.total_time:.2f}s  Tokens: {response.total_tokens}  Cost: ${response.total_cost:.6f}")
        print(f"\n  Step Breakdown:")
        for step_name, metrics in response.breakdown.items():
            model_str = metrics["model"] if metrics["model"] else "No API call"
            print(f"    {step_name:40s} | {metrics['time']:.3f}s | {model_str}")
        print()


if __name__ == "__main__":
    run_demo()