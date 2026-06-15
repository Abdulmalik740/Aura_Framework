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
    poetic: float
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
            "poetic": self.poetic,
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
        cook_type = context.get("cook_type", "home_cook")
        injury_mentioned = context.get("injury_mentioned", False)

        for rule in self._selection_rules:
            cond = rule["condition"]

            if cond == "injured":
                if emotion == "injured" or injury_mentioned:
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

            elif cond == "new_cook":
                if cook_type == "beginner" or context.get("product_knowledge") == "new_to_brand":
                    return rule["profile"]

            elif cond == "default":
                return rule["profile"]

        return "challenge"

    def _adjust_for_context(self, tone: ToneParameters, context: Dict, risk_flags: Dict) -> ToneParameters:
        adjusted = ToneParameters(**tone.__dict__)
        adj = self._adjustments

        # High urgency
        if context.get("urgency_level", 0) >= adj["high_urgency"]["urgency_threshold"]:
            adjusted.poetic = max(0.2, adjusted.poetic + adj["high_urgency"]["poetic_delta"])
            adjusted.brevity = adj["high_urgency"]["brevity"]
            adjusted.explanation_depth = adj["high_urgency"]["explanation_depth"]

        # Churn risk
        if risk_flags.get("churn_risk", 0) > adj["churn_risk"]["churn_risk_threshold"]:
            adjusted.empathy = min(1.0, adjusted.empathy + adj["churn_risk"]["empathy_delta"])
            adjusted.poetic = max(0.3, adjusted.poetic + adj["churn_risk"]["poetic_delta"])

        # Competitive cook
        if context.get("cook_type") in adj["competitive_cook"]["cook_types"]:
            adjusted.poetic = min(1.0, adjusted.poetic + adj["competitive_cook"]["poetic_delta"])
            adjusted.brevity = adj["competitive_cook"]["brevity"]

        # Time sensitive
        if context.get("time_sensitivity"):
            adjusted.brevity = adj["time_sensitive"]["brevity"]
            adjusted.explanation_depth = adj["time_sensitive"]["explanation_depth"]

        return adjusted

    def _enforce_boundaries(self, tone: ToneParameters) -> ToneParameters:
        b = self._boundaries
        adjusted = ToneParameters(**tone.__dict__)
        adjusted.poetic = max(b["min_poetic"], adjusted.poetic)
        adjusted.brevity = max(b["min_brevity"], adjusted.brevity)
        adjusted.use_contractions = b["use_contractions"]
        adjusted.use_imperatives = b["use_imperatives"]
        adjusted.avoid_softness = b["avoid_softness"]
        return adjusted

    def _smooth_transition(self, new_tone: ToneParameters) -> ToneParameters:
        max_shift = self._boundaries["max_tone_shift_per_turn"]
        adjusted = ToneParameters(**new_tone.__dict__)

        for attr in ("poetic", "empathy"):
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

        if tone_params.poetic > v["high_intensity_threshold"]:
            instructions.append(v["high_intensity_suffix"])
        elif tone_params.poetic < v["low_intensity_threshold"]:
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
EmberEdgeToneEngine = ToneAdaptationEngine