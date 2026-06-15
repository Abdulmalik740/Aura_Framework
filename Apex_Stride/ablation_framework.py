"""
Ablation Framework — EMNLP 2026 Industry Track
================================================
Rebuilds UniversalAdaptiveFramework with clean module separation.

Three independently togglable modules:
  - Context  : emotion/intent/athlete detection
  - Tone     : profile selection + generation instructions
  - BrandGuard: forbidden-phrase constraints + post-generation validation

System prompt = base_identity
              + [context layer   if use_context]
              + [tone layer      if use_tone]
              + [guard layer     if use_brand_guard]

This lets you run all 6 ablation conditions cleanly:
  A: {} — vanilla LLM
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
from context_understanding_engine import ContextUnderstandingEngine, ContextAnalysis, Intent, ConversationType
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
          "description": "No modules — raw gpt-4o-mini with brand name only"},
    "B": {"context": True,  "tone": False, "brand_guard": False,
          "label": "Context Only",
          "description": "Intent/emotion detection, no tone adaptation, no constraints"},
    "C": {"context": False, "tone": True,  "brand_guard": False,
          "label": "Tone Only",
          "description": "Fixed challenge profile, no context, no constraints"},
    "D": {"context": True,  "tone": True,  "brand_guard": False,
          "label": "Context + Tone",
          "description": "Adaptive tone from context, no brand guard"},
    "E": {"context": True,  "tone": False, "brand_guard": True,
          "label": "Context + Guard",
          "description": "Context-aware + forbidden-phrase constraints, no tone"},
    "F": {"context": True,  "tone": True,  "brand_guard": True,
          "label": "Full System",
          "description": "All three modules active"},
}

# ── null/default objects when modules are disabled ───────────────────────────
def _null_context() -> ContextAnalysis:
    """Returned when context module is disabled."""
    return ContextAnalysis(
        primary_emotion="neutral", emotion_intensity=2, sentiment_score=0.0,
        primary_intent=Intent.PRODUCT_QUESTION, confidence=0.5,
        conversation_type=ConversationType.PRODUCT_INQUIRY,
        user_state="neutral", urgency_level=2, frustration_level=1,
        motivation_level="medium", training_context=None,
        athlete_type="casual", formality_preference="casual",
        product_knowledge="familiar", technical_level="intermediate",
        situation_type="general_inquiry", key_pain_points=[],
        time_sensitivity=False, escalation_indicators=[],
        churn_indicators=[], competitor_mentions=[],
        injury_mentioned=False,
    )

def _default_tone_params() -> ToneParameters:
    """Fixed 'challenge' profile returned when tone module is disabled."""
    return ToneParameters(
        challenge_intensity=0.8, empathy=0.3, energy=0.7, brevity=0.9,
        vocabulary_style="challenge", explanation_depth="minimal",
    )


@dataclass
class AblationResponse:
    condition: str                   # "A" through "F"
    condition_label: str
    response_text: str
    context_used: bool
    tone_used: bool
    guard_used: bool
    context_analysis: Dict
    tone_params: Dict
    brand_validation: Dict           # {"passed": bool, "score": float, "violations": list}
    latency_s: float
    tokens: int
    cost: float
    # ── EMNLP metrics ─────────────────────────────────────────────────────
    third_metric_score:  float = 0.0  # bucket-specific rule-based 0-5
    swg:                 float = 0.0  # Severity-Weighted Guard 0-5
    third_metric_detail: Dict  = None
    swg_detail:          Dict  = None
    # ── Component latencies ms ────────────────────────────────────────────
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
    ) -> AblationResponse:
        """Run one ablation condition for a single message."""
        flags = ABLATION_CONDITIONS[condition]
        return self._process(
            message, session_id, user_id,
            use_context    = flags["context"],
            use_tone       = flags["tone"],
            use_brand_guard= flags["brand_guard"],
            condition      = condition,
            condition_label= flags["label"],
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
            # Each condition gets its own session so state doesn't bleed across
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
        lt = self.latency_tracker          # shorthand

        conv_state = self._get_or_create_session(session_id, user_id)
        turn_idx   = len(conv_state.turns)

        # ── Module 1: Context (Change 5: timed) ─────────────────────────────
        with lt.time("context"):
            if use_context:
                context = self.context_engine.analyze(
                    message=message,
                    history=conv_state.turns,
                    user_profile=conv_state.athlete_profile.__dict__,
                )
                total_tokens += getattr(context, "tokens_used", 0)
            else:
                context = _null_context()

        # ── Module 2: Tone (Change 5: timed + Change 4: consistency tracked) ─
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

        # Change 4: record tone for consistency tracking
        self.tone_tracker.record(session_id, turn_idx, tone)

        # ── Build prompt from active layers ──────────────────────────────────
        system_prompt = self._build_layered_prompt(
            message, context, tone, conv_state,
            use_context=use_context,
            use_tone=use_tone,
            use_brand_guard=use_brand_guard,
        )

        # ── Generate response (Change 5: timed) ──────────────────────────────
        with lt.time("generation"):
            response_text, gen_tokens = self._generate(system_prompt, message, conv_state)
        total_tokens += gen_tokens

        # ── Module 3: Brand Guard (Change 3: severity-weighted + Change 5: timed) ──
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
                # Change 3: severity-weighted score replaces flat score
                swg_result = severity_weighted_guard_score(raw_violations)
                brand_validation = {
                    "passed":     val.passed,
                    "score":      swg_result["score"],   # severity-weighted
                    "raw_score":  val.overall_score,     # original for comparison
                    "violations": raw_violations,
                    "swg_detail": swg_result,
                }
            else:
                brand_validation = {"passed": True, "score": 1.0,
                                    "raw_score": 1.0, "violations": [],
                                    "swg_detail": {}}

        # ── Compute bucket-aware third metric ────────────────────────────────
        intent_str = context.primary_intent.value
        new_metrics = compute_all_metrics(
            response=response_text,
            intent=intent_str,
            bucket=bucket,
            expected_contains=[],
            violations=brand_validation["violations"],
        )

        # ── State update ──────────────────────────────────────────────────────
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
            # ── new metrics ──────────────────────────────────────────────
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
    ) -> str:
        bc = self.brand_config
        layers: List[str] = []

        # ── Layer 0: Base identity (always present) ───────────────────────────
        products = cfg("products") or {}
        products_text = "\n".join(
            f"- {name}: best for {', '.join(p['best_for'])} | "
            f"features: {', '.join(p['features'])}"
            for name, p in products.items()
        )
        layers.append(f"""You are a {bc['name']} customer support agent.
Brand: {bc['personality']}
Tagline: {bc['tagline']} (never say this in responses)
Mission: {bc['mission']}

APPROVED PRODUCTS ONLY — never invent others:
{products_text}

Always recommend a specific product by name when relevant.""")

        # ── Layer 1: Context (use_context=True) ───────────────────────────────
        if use_context:
            intent_limits = {
                "complaint": 35, "product_issue": 35,
                "return_refund": 35, "replacement": 35,
                "motivation_seeking": 40, "training_question": 40,
                "product_question": 30, "sizing_help": 30,
                "purchase_intent": 30, "competition_comparison": 30,
                "praise": 20, "closing": 20,
            }
            word_limit = intent_limits.get(context.primary_intent.value, 40)

            intent_behaviour = {
                "complaint":            "Own it immediately ('That's on us.'). Resolution path next. No softening.",
                "product_issue":        "Own the failure. Ask for order number + photo. State resolution.",
                "return_refund":        "If complaint context: own it first, then process steps. If process question: steps only.",
                "replacement":          "Own it. Order number + photo → ship replacement. Tracking in 24h.",
                "motivation_seeking":   "One concrete physical action for today. No platitudes.",
                "training_question":    "2-3 punchy training principles. Redirect to relevant shoe. Max 40 words.",
                "product_question":     "Product name first. One reason. One imperative.",
                "sizing_help":          "Name the product. State sizing guidance directly.",
                "competition_comparison":"NEVER name any competitor brand. NEVER compare features, cushioning, or technology. Respond using brand values only. Under 25 words.",
                "purchase_intent":      "State availability + one fact. Done.",
                "praise":               "Brief, warm, move on. Under 20 words.",
                "closing":              "Brief acknowledgement. Under 20 words.",
            }
            behaviour = intent_behaviour.get(
                context.primary_intent.value,
                "Be direct. Answer the question. One imperative to close."
            )

            layers.append(f"""=== CONTEXT LAYER ===
Athlete state: emotion={context.primary_emotion} (intensity {context.emotion_intensity}/5),
               motivation={context.motivation_level}, type={context.athlete_type}
Intent detected: {context.primary_intent.value} (confidence {context.confidence:.0%})
Urgency: {context.urgency_level}/5 | Injury mentioned: {context.injury_mentioned}
Escalation signals: {', '.join(context.escalation_indicators) if context.escalation_indicators else 'none'}
Competitors mentioned: {', '.join(context.competitor_mentions) if context.competitor_mentions else 'none'}

WORD LIMIT for this intent: MAX {word_limit} words.
REQUIRED BEHAVIOUR: {behaviour}

ESCALATION RULE: If legal/sue/lawyer/bbb signals present → own product failure in
one sentence. State resolution path. Do NOT acknowledge the legal threat.""")

        # ── Layer 2: Tone (use_tone=True) ─────────────────────────────────────
        if use_tone:
            tone_instructions = self.tone_adapter.get_generation_instructions(
                tone, context.to_dict()
            )
            competition_note = ""
            if context.primary_intent.value == "competition_comparison":
                competition_note = "\nCOMPETITOR: Open with 'You vs. You.' or 'The only opponent is you.' Ultra-brief. Under 15 words total."
            layers.append(f"""=== TONE LAYER ===
Profile: {tone.vocabulary_style} | Challenge: {tone.challenge_intensity:.1f} | Empathy: {tone.empathy:.1f} | Brevity: {tone.brevity:.1f}
{tone_instructions}{competition_note}""")
        else:
            # Minimal style guidance when tone module is off — same for all conditions
            layers.append("""=== TONE LAYER (DEFAULT) ===
Be direct and concise. Use active language. Keep sentences short.""")

        # ── Layer 3: Brand Guard constraints (use_brand_guard=True) ──────────
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
  ✗ we're excited to announce | valued customer | discover your potential
  ✗ join the journey | you got this | push beyond | push your limits
  ✗ find what feels right | thank you for contacting
  ✗ we apologize for any inconvenience | is there anything else I can help
  ✗ crush it | you're amazing | starting strong | claiming your greatness

COMPETITOR NAMES (never mention):
  ✗ Nike | Adidas | Brooks | Hoka | Asics | New Balance | Saucony | Reebok

REQUIRED LANGUAGE PATTERNS:
  ✓ Contractions: you're, don't, can't, it's, we're
  ✓ Ownership: 'That's on us.' (not 'we apologize')
  ✓ Action verbs: Go. Push. Start. Move. Train. Earn.
  ✓ Sentence style: fragments OK, max ~10 words per sentence""")

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

        system_prompt = f"""You are {self.brand_config['name']} — regenerate a corrected response.

VIOLATIONS TO FIX:
{chr(10).join(issues) if issues else 'General brand voice violation — be more direct and punchy'}

User message: "{message}"
Athlete state: {context.primary_emotion}, {context.motivation_level} motivation
Intent: {context.primary_intent.value}

Constraints: short sentences, contractions, action verbs, no forbidden phrases.
Recommend only approved products: Runner 5, Velocity X2, Ultra Trail, Recovery Stride, Marathon Pro."""

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
        mp = pricing.get(model, {"prompt": 0.075, "completion": 0.3})
        blended = (mp["prompt"] + mp["completion"]) / 2
        return (tokens / 1_000_000) * blended
