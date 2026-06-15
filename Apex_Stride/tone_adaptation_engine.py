"""
Tone Adaptation Engine
All profiles, thresholds, and vocabulary loaded from brand_config.yaml.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI
from config_loader import (
    tone_profiles as get_profiles,
    tone_selection_rules as get_selection_rules,
    tone_adjustments as get_adjustments,
    tone_brand_boundaries as get_boundaries,
    tone_vocabulary as get_vocabulary,
)


@dataclass
class ToneParameters:
    challenge_intensity: float
    empathy: float
    energy: float
    brevity: float
    vocabulary_style: str
    explanation_depth: str
    use_contractions: bool = True
    use_imperatives: bool = True
    avoid_softness: bool = True

    def to_dict(self) -> Dict:
        return {
            "challenge_intensity": self.challenge_intensity,
            "empathy": self.empathy,
            "energy": self.energy,
            "brevity": self.brevity,
            "vocabulary_style": self.vocabulary_style,
            "explanation_depth": self.explanation_depth,
        }


class ToneAdaptationEngine:
    """
    Tone engine driven entirely by brand_config.yaml.
    No profiles, thresholds, or vocabulary hardcoded here.
    """

    def __init__(self, brand_boundaries: Dict, client: OpenAI):
        self.brand_boundaries = brand_boundaries
        self.client = client
        self._profiles = get_profiles()
        self._selection_rules = get_selection_rules()
        self._adjustments = get_adjustments()
        self._boundaries = get_boundaries()
        self._vocab = get_vocabulary()
        self.last_tone: Optional[ToneParameters] = None

    def _profile_to_params(self, name: str) -> ToneParameters:
        p = self._profiles[name]
        return ToneParameters(**p, use_contractions=True, use_imperatives=True, avoid_softness=True)

    def select_tone(self, context: Dict, conversation_type: str, risk_flags: Dict) -> ToneParameters:
        profile_name = self._select_profile(context, risk_flags)
        tone = self._profile_to_params(profile_name)
        tone = self._adjust_for_context(tone, context, risk_flags)
        tone = self._enforce_boundaries(tone)
        if self.last_tone:
            tone = self._smooth_transition(tone)
        self.last_tone = tone
        return tone

    def _select_profile(self, context: Dict, risk_flags: Dict) -> str:
        emotion = context.get("emotion", "neutral")
        intent = context.get("intent", "")
        motivation = context.get("motivation_level", "medium")
        athlete_type = context.get("athlete_type", "casual")
        injury_mentioned = context.get("injury_mentioned", False)
        escalation = context.get("escalation_indicators") or []

        # Service intents that require coach tone regardless of injury status.
        # When athlete is injured AND has a complaint/refund/replacement intent,
        # the brand needs to own the issue (coach), not just empathise (support).
        # Escalation signals also bypass the injured rule — LLM handles the
        # injury acknowledgment via the escalation behaviour rule in the prompt.
        _service_intents = {"product_issue", "complaint", "return_refund",
                            "replacement", "order_inquiry"}

        for rule in self._selection_rules:
            cond = rule["condition"]

            if cond == "injured":
                if emotion == "injured" or injury_mentioned:
                    # Skip injured→support when intent is service or escalation present
                    if intent in _service_intents or escalation:
                        continue
                    return rule["profile"]

            elif cond == "vulnerable_motivation":
                vulnerable = rule.get("vulnerable_emotions", [])
                if (intent == "motivation_seeking" or motivation == "low") and emotion in vulnerable:
                    return rule["profile"]

            elif cond == "motivation_needed":
                if intent == "motivation_seeking" or motivation == "low":
                    return rule["profile"]

            elif cond == "service_issue":
                service_intents = rule.get("service_intents", [])
                angry_emotions = rule.get("angry_emotions", [])
                urgency_threshold = rule.get("urgency_threshold", 4)
                if intent in service_intents:
                    if emotion in angry_emotions or context.get("urgency_level", 0) >= urgency_threshold:
                        return rule["profile"]

            elif cond == "new_athlete":
                if athlete_type == "beginner" or context.get("product_knowledge") == "new_to_brand":
                    return rule["profile"]

            elif cond == "default":
                return rule["profile"]

        return "challenge"

    def _adjust_for_context(self, tone: ToneParameters, context: Dict, risk_flags: Dict) -> ToneParameters:
        adjusted = ToneParameters(**tone.__dict__)
        adj = self._adjustments

        # High urgency
        if context.get("urgency_level", 0) >= adj["high_urgency"]["urgency_threshold"]:
            adjusted.challenge_intensity = max(0.2, adjusted.challenge_intensity + adj["high_urgency"]["challenge_intensity_delta"])
            adjusted.brevity = adj["high_urgency"]["brevity"]
            adjusted.explanation_depth = adj["high_urgency"]["explanation_depth"]

        # Churn risk
        if risk_flags.get("churn_risk", 0) > adj["churn_risk"]["churn_risk_threshold"]:
            adjusted.empathy = min(1.0, adjusted.empathy + adj["churn_risk"]["empathy_delta"])
            adjusted.challenge_intensity = max(0.3, adjusted.challenge_intensity + adj["churn_risk"]["challenge_intensity_delta"])

        # Competitive athlete
        if context.get("athlete_type") in adj["competitive_athlete"]["athlete_types"]:
            adjusted.challenge_intensity = min(1.0, adjusted.challenge_intensity + adj["competitive_athlete"]["challenge_intensity_delta"])
            adjusted.brevity = adj["competitive_athlete"]["brevity"]

        # Time sensitive
        if context.get("time_sensitivity"):
            adjusted.brevity = adj["time_sensitive"]["brevity"]
            adjusted.explanation_depth = adj["time_sensitive"]["explanation_depth"]

        return adjusted

    def _enforce_boundaries(self, tone: ToneParameters) -> ToneParameters:
        b = self._boundaries
        adjusted = ToneParameters(**tone.__dict__)
        adjusted.challenge_intensity = max(b["min_challenge_intensity"], adjusted.challenge_intensity)
        adjusted.brevity = max(b["min_brevity"], adjusted.brevity)
        adjusted.use_contractions = b["use_contractions"]
        adjusted.use_imperatives = b["use_imperatives"]
        adjusted.avoid_softness = b["avoid_softness"]
        return adjusted

    def _smooth_transition(self, new_tone: ToneParameters) -> ToneParameters:
        max_shift = self._boundaries["max_tone_shift_per_turn"]
        adjusted = ToneParameters(**new_tone.__dict__)

        for attr in ("challenge_intensity", "empathy"):
            diff = getattr(new_tone, attr) - getattr(self.last_tone, attr)
            if abs(diff) > max_shift:
                setattr(adjusted, attr, getattr(self.last_tone, attr) + (max_shift if diff > 0 else -max_shift))

        return adjusted

    def get_generation_instructions(self, tone_params: ToneParameters, context: Dict) -> str:
        v = self._vocab
        instructions = [v["base"]]

        style = tone_params.vocabulary_style
        if style in v:
            instructions.append(v[style])

        if tone_params.challenge_intensity > v["high_intensity_threshold"]:
            instructions.append(v["high_intensity_suffix"])
        elif tone_params.challenge_intensity < v["low_intensity_threshold"]:
            instructions.append(v["low_intensity_suffix"])

        if tone_params.empathy > v["high_empathy_threshold"]:
            instructions.append(v["high_empathy_suffix"])

        if tone_params.brevity > v["ultra_brief_threshold"]:
            instructions.append(v["ultra_brief_suffix"])
        elif tone_params.brevity > v["concise_threshold"]:
            instructions.append(v["concise_suffix"])

        if tone_params.explanation_depth == "tactical":
            instructions.append(v["tactical_suffix"])
        elif tone_params.explanation_depth == "minimal":
            instructions.append(v["minimal_suffix"])

        instructions.append(v["forbidden_never_use"])
        return " ".join(instructions)


# Backwards-compatible alias
ApexStrideToneEngine = ToneAdaptationEngine