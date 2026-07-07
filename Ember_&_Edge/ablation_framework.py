"""
Ablation Framework — Ember & Edge 
==============================================================
Rebuilds UniversalAdaptiveFramework with clean module separation.

Three independently togglable modules:
  - Context    : emotion / intent / cook-type detection
  - Tone       : profile selection + generation instructions
  - BrandGuard : forbidden-phrase constraints + post-generation validation

System prompt = base_identity
              + [context layer    if use_context]
              + [tone layer       if use_tone]
              + [guard layer      if use_brand_guard]

This lets you run all 6 ablation conditions cleanly:
  A: {}                       — vanilla LLM
  B: {context}
  C: {tone}
  D: {context, tone}
  E: {context, brand_guard}
  F: {context, tone, brand_guard} — full system
"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from context_understanding_engine import (
    ContextUnderstandingEngine, ContextAnalysis, Intent, ConversationType,
)
from tone_adaptation_engine import ToneAdaptationEngine, ToneParameters
from brand_consistency_guard import BrandConsistencyGuard
from conversation_state_manager import ConversationStateManager
from config_loader import build_brand_ethos, llm as get_llm, cfg

# ── EMNLP instrumentation + metrics ──────────────────────────────────────────
from instrumentation import (
    ContextEngineTracker,
    ToneConsistencyTracker,
    ComponentLatencyTracker,
)
from metrics import (
    severity_weighted_guard_score,
    compute_all_metrics,
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── ablation condition registry ───────────────────────────────────────────────
ABLATION_CONDITIONS: Dict[str, Dict] = {
    "A": {"context": False, "tone": False, "brand_guard": False,
          "label": "Vanilla LLM",
          "description": "No modules — raw gpt-4o with brand name only"},
    "B": {"context": True,  "tone": False, "brand_guard": False,
          "label": "Context Only",
          "description": "Intent/emotion/cook-type detection, no tone adaptation, no constraints"},
    "C": {"context": False, "tone": True,  "brand_guard": False,
          "label": "Tone Only",
          "description": "Fixed sensory profile, no context, no constraints"},
    "D": {"context": True,  "tone": True,  "brand_guard": False,
          "label": "Context + Tone",
          "description": "Adaptive tone from context, no brand guard"},
    "E": {"context": True,  "tone": False, "brand_guard": True,
          "label": "Context + Guard",
          "description": "Context-aware + forbidden-phrase constraints, no tone"},
    "F": {"context": True,  "tone": True,  "brand_guard": True,
          "label": "Full System",
          "description": "All three modules active"},
    # ── Baselines ────────────────────────────────────────────────────────────
    "G": {"context": False, "tone": False, "brand_guard": False,
          "label": "Prompt-Only Baseline",
          "description": "Brand name + product list only. No personality, no modules. Absolute floor."},
    "H": {"context": False, "tone": False, "brand_guard": False,
          "label": "Few-Shot Baseline",
          "description": "6 hand-crafted examples (one per bucket) in system prompt. No modules."},
}

# ── Few-shot examples for condition H (one ideal response per bucket) ─────────
_FEW_SHOT_EXAMPLES = [
    {
        "bucket": "complaint",
        "user": "My knife chipped on the very first use.",
        "assistant": "That's not right. Send us a photo — we'll replace it today.",
    },
    {
        "bucket": "escalation",
        "user": "This knife is dangerous. It slipped and I cut myself badly.",
        "assistant": "That shouldn't happen. We're sending a replacement today. Hope you heal quickly.",
    },
    {
        "bucket": "competitor",
        "user": "Why should I buy from you instead of Wusthof?",
        "assistant": "History is theirs. This moment is yours. The Artisan Chef's Knife — available now. What do you cook most?",
    },
    {
        "bucket": "product_question",
        "user": "I've never owned a decent knife. Where do I start?",
        "assistant": "Start with the Beginner Set. Two knives, everything you need. What do you find yourself cooking most?",
    },
    {
        "bucket": "purchase_intent",
        "user": "I want to order the Artisan Chef's Knife. Is it in stock?",
        "assistant": "Yes, in stock. The Artisan Chef's Knife is ready when you are.",
    },
    {
        "bucket": "edge_case",
        "user": "I want a refund and I'm going to sue.",
        "assistant": "That's not right. Refund processed immediately — send us the details.",
    },
]


# ── null/default objects when modules are disabled ───────────────────────────
def _null_context() -> ContextAnalysis:
    """Returned when context module is disabled."""
    return ContextAnalysis(
        primary_emotion="neutral", emotion_intensity=2, sentiment_score=0.0,
        primary_intent=Intent.PRODUCT_QUESTION, confidence=0.5,
        conversation_type=ConversationType.PRODUCT_INQUIRY,
        user_state="neutral", urgency_level=2, frustration_level=1,
        motivation_level="medium", training_context=None,
        cook_type="home_cook", formality_preference="casual",
        product_knowledge="new_to_brand", technical_level="beginner",
        situation_type="general_inquiry", key_pain_points=[],
        time_sensitivity=False, escalation_indicators=[],
        churn_indicators=[], competitor_mentions=[],
        injury_mentioned=False,
    )


def _default_tone_params() -> ToneParameters:
    """Fixed 'sensory' profile returned when tone module is disabled."""
    return ToneParameters(
        poetic=0.3, empathy=0.6, energy=0.5, brevity=0.6,
        vocabulary_style="sensory", explanation_depth="standard",
    )


@dataclass
class AblationResponse:
    condition: str                    # "A" through "F"
    condition_label: str
    response_text: str
    context_used: bool
    tone_used: bool
    guard_used: bool
    context_analysis: Dict
    tone_params: Dict
    brand_validation: Dict            # {"passed": bool, "score": float, "violations": list}
    latency_s: float
    tokens: int
    cost: float
    # ── EMNLP metrics ──────────────────────────────────────────────────────
    third_metric_score:  float = 0.0  # bucket-specific rule-based 0-5
    swg:                 float = 0.0  # Severity-Weighted Guard 0-5
    third_metric_detail: Dict  = None
    swg_detail:          Dict  = None
    # ── Component latencies ms ─────────────────────────────────────────────
    component_latencies: Dict = None  # {context, tone, generation, guard, total}


class AblationFramework:
    """
    Drop-in replacement for UniversalAdaptiveFramework that supports
    per-condition module toggling for ablation studies.
    """

    def __init__(self, client: OpenAI):
        self.client = client
        self._llm = get_llm()
        self.brand_config = build_brand_ethos()

        brand_boundaries = {
            "core_values":       self.brand_config["core_values"],
            "messaging_pillars": self.brand_config["messaging_pillars"],
            "voice_guidelines":  self.brand_config["voice_guidelines"],
            "personality":       self.brand_config["personality"],
        }
        self.context_engine = ContextUnderstandingEngine(client)
        self.tone_adapter   = ToneAdaptationEngine(brand_boundaries, client)
        self.brand_guard    = BrandConsistencyGuard(self.brand_config, client)
        self.active_sessions: Dict[str, ConversationStateManager] = {}

        # ── EMNLP instrumentation trackers ────────────────────────────────
        self.ctx_tracker     = ContextEngineTracker()
        self.tone_tracker    = ToneConsistencyTracker()
        self.latency_tracker = ComponentLatencyTracker()

        # Wire context tracker onto engine without modifying engine source
        _orig_analyze = self.context_engine.analyze
        _tracker = self.ctx_tracker
        def _instrumented_analyze(message, history=None, user_profile=None):
            return _tracker.track(_orig_analyze, message, history, user_profile)
        self.context_engine.analyze = _instrumented_analyze

    # ── public API ────────────────────────────────────────────────────────────

    def run_condition(
        self,
        message: str,
        session_id: str,
        condition: str,                        # "A" … "F"
        user_id: Optional[str] = None,
        bucket: str = "default",
    ) -> AblationResponse:
        """Run one ablation condition for a single message."""
        flags = ABLATION_CONDITIONS[condition]
        return self._process(
            message, session_id, user_id,
            use_context     = flags["context"],
            use_tone        = flags["tone"],
            use_brand_guard = flags["brand_guard"],
            condition       = condition,
            condition_label = flags["label"],
            bucket          = bucket,
        )

    def run_all_conditions(
        self,
        message: str,
        base_session_id: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, AblationResponse]:
        """Run all 6 ablation conditions for the same message."""
        results = {}
        for cond in ABLATION_CONDITIONS:
            sid = f"{base_session_id}_{cond}"
            results[cond] = self.run_condition(message, sid, cond, user_id)
        return results

    # ── core processing ───────────────────────────────────────────────────────

    def _process(
        self,
        message: str,
        session_id: str,
        user_id: Optional[str],
        use_context: bool,
        use_tone: bool,
        use_brand_guard: bool,
        condition: str,
        condition_label: str,
        bucket: str = "default",
    ) -> AblationResponse:
        t0 = time.time()
        total_tokens = 0
        lt = self.latency_tracker

        conv_state = self._get_or_create_session(session_id, user_id)
        turn_idx   = len(conv_state.turns)

        # ── Module 1: Context (timed) ──────────────────────────────────────
        with lt.time("context"):
            if use_context:
                context = self.context_engine.analyze(
                    message=message,
                    history=conv_state.turns,
                    user_profile=conv_state.cook_profile.__dict__,
                )
                total_tokens += getattr(context, "tokens_used", 0)
            else:
                context = _null_context()

        # ── Module 2: Tone (timed + consistency tracked) ───────────────────
        with lt.time("tone"):
            if use_tone:
                tone = self.tone_adapter.select_tone(
                    context=context.to_dict(),
                    conversation_type=context.conversation_type.value,
                    risk_flags={
                        "churn_risk":            conv_state.churn_risk,
                        "competitor_switch_risk": conv_state.competitor_switch_risk,
                    },
                )
            else:
                tone = _default_tone_params()

        # Record tone for consistency tracking
        self.tone_tracker.record(session_id, turn_idx, tone)

        # ── Build prompt from active layers ───────────────────────────────
        system_prompt = self._build_layered_prompt(
            message, context, tone, conv_state,
            use_context=use_context,
            use_tone=use_tone,
            use_brand_guard=use_brand_guard,
            condition=condition,
        )

        # ── Generate response (timed) ──────────────────────────────────────
        with lt.time("generation"):
            response_text, gen_tokens = self._generate(system_prompt, message, conv_state)
        total_tokens += gen_tokens

        # ── Module 3: Brand Guard (severity-weighted + timed) ─────────────
        with lt.time("guard"):
            if use_brand_guard:
                val = self.brand_guard.validate(
                    response_text,
                    context=context.to_dict(),
                    conversation_type=conv_state.get_conversation_type_for_guard(),
                )
                if not val.passed:
                    response_text, regen_tokens = self._regenerate_with_fixes(
                        message, context, tone, val, conv_state, use_brand_guard=True
                    )
                    total_tokens += regen_tokens
                    val = self.brand_guard.validate(response_text, context=context.to_dict())

                raw_violations = [
                    {"severity": v.severity, "category": v.category,
                     "description": v.description}
                    for v in val.violations
                ]
                swg_result = severity_weighted_guard_score(raw_violations)
                brand_validation = {
                    "passed":     val.passed,
                    "score":      swg_result["score"],
                    "raw_score":  val.overall_score,
                    "violations": raw_violations,
                    "swg_detail": swg_result,
                }
            else:
                brand_validation = {"passed": True, "score": 1.0,
                                    "raw_score": 1.0, "violations": [],
                                    "swg_detail": {}}

        # ── Compute bucket-aware third metric ──────────────────────────────
        intent_str = context.primary_intent.value
        new_metrics = compute_all_metrics(
            response=response_text,
            intent=intent_str,
            bucket=bucket,
            expected_contains=[],
            violations=brand_validation["violations"],
            original_message=message,  # ADD THIS LINE
        )
                

        # ── State update ───────────────────────────────────────────────────
        conv_state.add_turn(message, response_text, context.to_dict(), tone.to_dict())

        return AblationResponse(
            condition=condition,
            condition_label=condition_label,
            response_text=response_text,
            context_used=use_context,
            tone_used=use_tone,
            guard_used=use_brand_guard,
            context_analysis=context.to_dict(),
            tone_params=tone.to_dict(),
            brand_validation=brand_validation,
            latency_s=round(time.time() - t0, 3),
            tokens=total_tokens,
            cost=self._calc_cost(self._llm["generation_model"], total_tokens),
            third_metric_score=new_metrics["third_metric_score"],
            swg=new_metrics["swg"],
            third_metric_detail=new_metrics["third_metric_detail"],
            swg_detail=new_metrics["swg_detail"],
            component_latencies=lt.last(),
        )

    # ── 4-layer prompt builder ────────────────────────────────────────────────

    def _build_layered_prompt(
        self,
        message: str,
        context: ContextAnalysis,
        tone: ToneParameters,
        conv_state: ConversationStateManager,
        use_context: bool,
        use_tone: bool,
        use_brand_guard: bool,
        condition: str = "F",
    ) -> str:
        bc = self.brand_config
        layers: List[str] = []

        # ── Baseline G: prompt-only (brand name + products, nothing else) ──
        if condition == "G":
            products = cfg("products") or {}
            products_text = "\n".join(
                f"- {name}: best for {', '.join(p['best_for'])}"
                for name, p in products.items()
            )
            return f"""You are a customer service assistant for {bc['name']}.
Available products:
{products_text}
Answer the customer's question helpfully."""

        # ── Baseline H: few-shot (examples only, no modules) ─────────────
        if condition == "H":
            products = cfg("products") or {}
            products_text = "\n".join(
                f"- {name}: best for {', '.join(p['best_for'])}"
                for name, p in products.items()
            )
            examples_text = "\n\n".join(
                f"User: {ex['user']}\nAssistant: {ex['assistant']}"
                for ex in _FEW_SHOT_EXAMPLES
            )
            return f"""You are a customer service assistant for {bc['name']} — a sensory, poetic knife brand.

Here are examples of ideal responses:

{examples_text}

Available products:
{products_text}

Match the tone and style of the examples above. Be concise and direct."""

        # ── Layer 0: Base identity (always present) ────────────────────────
        products = cfg("products") or {}
        products_text = "\n".join(
            f"- {name}: best for {', '.join(p['best_for'])} | "
            f"cook types: {', '.join(p['cook_types'])}"
            for name, p in products.items()
        )
        layers.append(f"""You are an {bc['name']} customer guide — a patient chef-instructor.
Brand: {bc['personality']}
Tagline: "{bc['tagline']}" — never say this in responses.
Mission: {bc['mission']}

APPROVED PRODUCTS ONLY — never invent others:
{products_text}

Always recommend a specific product by name when relevant.""")

        # ── Layer 1: Context (use_context=True) ────────────────────────────
        if use_context:
            intent_limits = {
                "complaint": 60, "product_issue": 60,
                "return_refund": 60, "replacement": 60,
                "fear_concern": 70, "sharpening_help": 70,
                "care_question": 70, "product_question": 65,
                "comparison": 50, "purchase_intent": 40,
                "price_inquiry": 40, "technical_question": 70,
                "praise": 30, "closing": 25,
            }
            word_limit = intent_limits.get(context.primary_intent.value, 65)

            intent_behaviour = {
                # NOTE: Do NOT put ownership phrases here ("That's not right." etc).
                # Those only belong in the Brand Guard layer. If they appear here,
                # condition B (context only, no guard) will score AC=5.0 incorrectly,
                # masking the guard module's contribution.
                "complaint":          "Open with 'That's not right.' or 'That shouldn't happen.' — FIRST words. State the concrete fix next. No poetry, no sensory language. No 'safety is our priority'.",
                "product_issue":      "Open with 'That's not right.' or 'That shouldn't happen.' — FIRST words. State concrete resolution immediately. No sensory language. No 'safety is our priority'.",
                "return_refund":      "Open with 'That's not right.' or 'That shouldn't happen.' — FIRST words. Give the resolution path clearly. No poetry.",
                "replacement":        "Open with 'That's not right.' or 'That shouldn't happen.' — FIRST words. State resolution path immediately.",
                "fear_concern":       "Calm, reassuring. Recommend Paring Knife — smaller blade builds confidence. No sensory immersion.",
                "sharpening_help":    "One concrete maintenance step. Name the action. Gentle, practical.",
                "care_question":      "One or two care instructions. Direct and tactile. 'After each use: warm water, soft cloth, dry immediately.'",
                "product_question":   f"{'Name the Artisan Chef knife in the FIRST 5 WORDS. ONE craft word (forged/balance/weight/glide). One performance line. Decisive close. NO poetry.' if context.cook_type in ('professional_chef', 'competitive_cook') else ('Name the Beginner Set FIRST. MUST include ONE craft word: forged/balance/weight/glide. One reason it fits a beginner. One gentle question about what they cook.' if context.cook_type == 'beginner' else 'Lead with a gentle question about what they cook. Name the specific knife within first 20 words. ONE craft word (forged/balance/glide/whisper). One gentle close.')}",
                "comparison":         "Redirect to craft and feel. 'The knife serves the ingredient, not the comparison.' No feature specs.",
                "purchase_intent":    "Name the knife. State availability. One gentle close. Under 40 words.",
                "price_inquiry":      "State the price clearly. One sentence on value through feel, not specs.",
                "technical_question": "Redirect from specs to performance feel. Never quote HRC or blade angles as selling points.",
                "praise":             "Warm, brief acknowledgment. 2 sentences max.",
                "closing":            "Brief, graceful close. 1-2 sentences.",
            }
            behaviour = intent_behaviour.get(
                context.primary_intent.value,
                "Sensory and patient. Redirect to the ingredient. Ask one gentle question."
            )

            layers.append(f"""=== CONTEXT LAYER ===
Cook state: emotion={context.primary_emotion} (intensity {context.emotion_intensity}/5),
            motivation={context.motivation_level}, cook_type={context.cook_type}
Intent detected: {context.primary_intent.value} (confidence {context.confidence:.0%})
Urgency: {context.urgency_level}/5 | Injury/fear mentioned: {context.injury_mentioned}
Escalation signals: {', '.join(context.escalation_indicators) if context.escalation_indicators else 'none'}
Competitors mentioned: {', '.join(context.competitor_mentions) if context.competitor_mentions else 'none'}

WORD LIMIT for this intent: MAX {word_limit} words.
REQUIRED BEHAVIOUR: {behaviour}

COMPLAINT RULE: If complaint/product_issue detected → acknowledge directly and briefly,
state the fix immediately. Do NOT use poetry or sensory language.
(Specific ownership phrasing is enforced by the Brand Guard layer when active.)
FEAR RULE: If fear/injury detected → calm and practical, Paring Knife first,
DO NOT ask 'What do you feel?' — they haven't held the knife yet.""")

        # ── Layer 2: Tone (use_tone=True) ──────────────────────────────────
        if use_tone:
            tone_instructions = self.tone_adapter.get_generation_instructions(
                tone, context.to_dict()
            )
            comparison_note = ""
            if context.primary_intent.value == "comparison":
                comparison_note = "\nCOMPARISON: Open with craft philosophy ('The knife serves the ingredient, not the comparison.'). Never name competitors or spec compare. Under 20 words total."
            layers.append(f"""=== TONE LAYER ===
Profile: {tone.vocabulary_style} | Poetic: {tone.poetic:.1f} | Empathy: {tone.empathy:.1f} | Brevity: {tone.brevity:.1f}
{tone_instructions}{comparison_note}""")
        else:
            layers.append("""=== TONE LAYER (DEFAULT) ===
Natural flowing sentences. Patient and unhurried. Ask one gentle question where appropriate.""")

        # ── Layer 3: Brand Guard constraints (use_brand_guard=True) ───────
        if use_brand_guard:
            core_values = cfg("core_values") or []
            values_text = " | ".join(v["name"] for v in core_values)
            pillars = cfg("messaging_pillars") or []
            pillars_text = " | ".join(p["text"] for p in pillars)

            layers.append(f"""=== BRAND GUARD LAYER ===
CORE VALUES (apply all): {values_text}
MESSAGING PILLARS: {pillars_text}

VOICE — DO:
{chr(10).join('  ✓ ' + r for r in bc['voice_guidelines']['do'])}

VOICE — NEVER:
{chr(10).join('  ✗ ' + r for r in bc['voice_guidelines']['dont'])}

HARD FORBIDDEN PHRASES (instant failure):
  ✗ best-in-class | industry-leading | revolutionary | game-changing | premium quality
  ✗ innovative technology | synergy | leverage | buy now | don't miss out | limited time
  ✗ act now | hurry | does that make sense | does that help | let me know if | feel free to
  ✗ thank you for contacting | we apologize for any inconvenience | is there anything else
  ✗ excellent choice | let's get started | you've got this | amazing journey
  ✗ unlock your potential | level up your cooking

TECHNICAL SPECS — NEVER MENTION as selling points:
  ✗ HRC | Rockwell | blade angle | degrees | steel composition | geometry spec

COMPETITOR NAMES (never mention):
  ✗ Wusthof | Shun | Global | Victorinox | Henckels | Miyabi

COMPLAINT CONTEXT — REQUIRED:
  ✓ Acknowledge directly: 'That's not right.' or 'That shouldn't happen.'
  ✓ State concrete fix in the next sentence
  ✗ NEVER use poetry or sensory language in complaint responses

PRODUCT RECOMMENDATION:
  ✓ Natural flowing sentences (8-15 words average)
  ✓ Lead with sensory questions: feel, weight, balance, glide
  ✓ Redirect to the ingredient before the knife

LANGUAGE STYLE — REQUIRED:
  ✓ USE contractions throughout: it's / that's / you're / don't / can't / we'll / I'll
  ✗ NEVER write: "it is" / "that is" / "you are" / "do not" / "cannot" / "we will"
  ✓ SHORT sentences: aim for 8-12 words each. Fragments are fine.
  ✓ Example: "That's not right. Send us a photo — we'll fix it today." NOT
    "We are sorry to hear that you have experienced this issue with your knife." """)

        layers.append("=== YOUR RESPONSE ===\nRespond now. Follow all active layers above.")
        return "\n\n".join(layers)

    # ── generation helpers ────────────────────────────────────────────────────

    def _generate(
        self,
        system_prompt: str,
        message: str,
        conv_state: ConversationStateManager,
    ):
        messages = [{"role": "system", "content": system_prompt}]
        max_hist = self._llm.get("context_history_turns", 3)
        for turn in conv_state.turns[-max_hist:]:
            messages.append({"role": "user",      "content": turn.user_message})
            messages.append({"role": "assistant",  "content": turn.assistant_response})
        messages.append({"role": "user", "content": message})

        resp = self.client.chat.completions.create(
            model=self._llm["generation_model"],
            messages=messages,
            temperature=self._llm["generation_temperature"],
            max_tokens=self._llm["generation_max_tokens"],
        )
        tokens = resp.usage.total_tokens if hasattr(resp, "usage") else 0
        return resp.choices[0].message.content.strip(), tokens

    def _regenerate_with_fixes(
        self,
        message: str,
        context: ContextAnalysis,
        tone: ToneParameters,
        validation_result,
        conv_state: ConversationStateManager,
        use_brand_guard: bool,
    ):
        violations = getattr(validation_result, "violations", [])
        issues = []
        for v in violations[:3]:
            issues.append(f"  VIOLATION ({v.severity}): {v.description}")
            if v.suggestion:
                issues.append(f"  FIX: {v.suggestion}")

        system_prompt = f"""You are an {self.brand_config['name']} customer guide — regenerate a corrected response.

VIOLATIONS TO FIX:
{chr(10).join(issues) if issues else 'General brand voice violation — too corporate or too technical.'}

User message: "{message}"
Cook state: {context.primary_emotion}, {context.motivation_level} motivation
Intent: {context.primary_intent.value}

Rules: natural flowing sentences (8-15 words), no forbidden phrases,
no technical specs (HRC, angles), no poetry in complaints,
name a specific approved knife when recommending.
Approved products: Artisan Chef's Knife, Paring Knife, Bread Knife, Santoku, Beginner Set."""

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": message})

        resp = self.client.chat.completions.create(
            model=self._llm.get("regeneration_model", self._llm["generation_model"]),
            messages=messages,
            temperature=self._llm.get("regeneration_temperature", 0.3),
            max_tokens=self._llm.get("regeneration_max_tokens", 120),
        )
        tokens = resp.usage.total_tokens if hasattr(resp, "usage") else 0
        return resp.choices[0].message.content.strip(), tokens

    def _get_or_create_session(
        self, session_id: str, user_id: Optional[str]
    ) -> ConversationStateManager:
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = ConversationStateManager(
                session_id, user_id
            )
        return self.active_sessions[session_id]

    def _calc_cost(self, model: str, tokens: int) -> float:
        pricing = self._llm.get("pricing", {})
        mp = pricing.get(model, {"prompt": 2.50, "completion": 10.00})
        blended = (mp["prompt"] + mp["completion"]) / 2
        return (tokens / 1_000_000) * blended
