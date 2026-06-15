"""
Context Understanding Engine - Hybrid (rule-based + AI fallback)
All keywords, thresholds, mappings, and schema loaded from brand_config.yaml.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json
from openai import OpenAI
from config_loader import context_engine_config, llm


class Intent(Enum):
    PRODUCT_ISSUE = "product_issue"
    ORDER_INQUIRY = "order_inquiry"
    PRODUCT_QUESTION = "product_question"
    COMPLAINT = "complaint"
    RETURN_REFUND = "return_refund"
    REPLACEMENT = "replacement"
    PRAISE = "praise"
    MOTIVATION_SEEKING = "motivation_seeking"
    TRAINING_QUESTION = "training_question"
    SIZING_HELP = "sizing_help"
    COMPETITION_COMPARISON = "competition_comparison"
    PURCHASE = "purchase_intent"
    CLOSING = "closing"


class ConversationType(Enum):
    CUSTOMER_SUPPORT = "customer_support"
    PRODUCT_INQUIRY = "product_inquiry"
    COMPLAINT_RESOLUTION = "complaint_resolution"
    ATHLETE_MOTIVATION = "athlete_motivation"
    COMMUNITY_ENGAGEMENT = "community_engagement"


@dataclass
class ContextAnalysis:
    primary_emotion: str
    emotion_intensity: int
    sentiment_score: float
    primary_intent: Intent
    confidence: float
    conversation_type: ConversationType
    user_state: str
    urgency_level: int
    frustration_level: int
    motivation_level: str
    training_context: Optional[str]
    athlete_type: str
    formality_preference: str
    product_knowledge: str
    technical_level: str
    situation_type: str
    key_pain_points: List[str]
    time_sensitivity: bool
    escalation_indicators: List[str]
    churn_indicators: List[str]
    competitor_mentions: List[str]
    injury_mentioned: bool
    tokens_used: int = 0
    used_ai: bool = False
    secondary_emotions: List[str] = None
    multi_intent: bool = False
    intent_details: Dict = None
    directness: float = 0.7
    verbosity: str = "brief"
    inferred_age_group: Optional[str] = None
    age_confidence: str = "low"
    age_indicators: List[str] = None
    mentioned_products: List[str] = None
    satisfaction_indicators: List[str] = None
    training_goals_mentioned: bool = False

    def __post_init__(self):
        if self.secondary_emotions is None: self.secondary_emotions = []
        if self.intent_details is None: self.intent_details = {}
        if self.age_indicators is None: self.age_indicators = []
        if self.mentioned_products is None: self.mentioned_products = []
        if self.satisfaction_indicators is None: self.satisfaction_indicators = []
        self.training_goals_mentioned = self.training_context is not None

    def to_dict(self) -> Dict:
        return {
            "emotion": self.primary_emotion,
            "emotion_intensity": self.emotion_intensity,
            "sentiment_score": self.sentiment_score,
            "intent": self.primary_intent.value,
            "intent_confidence": self.confidence,
            "conversation_type": self.conversation_type.value,
            "user_state": self.user_state,
            "urgency_level": self.urgency_level,
            "motivation_level": self.motivation_level,
            "athlete_type": self.athlete_type,
            "training_context": self.training_context,
            "injury_mentioned": self.injury_mentioned,
            "situation_type": self.situation_type,
            "tokens_used": self.tokens_used,
            "used_ai": self.used_ai,
        }


class ContextUnderstandingEngine:
    """
    Hybrid context engine.
    All keywords, thresholds, and schema come from brand_config.yaml.
    """

    def __init__(self, client: OpenAI):
        self.client = client
        self._cfg = context_engine_config()
        self._llm = llm()
        self._ai_threshold = self._cfg["ai_fallback_threshold"]

    def analyze(self, message: str, history: Optional[List] = None, user_profile: Optional[Dict] = None) -> ContextAnalysis:
        msg_lower = message.lower()
        emotion, intensity, emotion_matched = self._detect_emotion(msg_lower, message)
        intent, confidence, intent_matched = self._detect_intent(msg_lower, message, emotion)

        should_use_ai = confidence < self._ai_threshold or (not emotion_matched and not intent_matched)
        if should_use_ai:
            return self._ai_analyze(message, history, user_profile)

        athlete_type = self._detect_athlete_type(msg_lower)
        motivation_level = self._detect_motivation_level(msg_lower, emotion, history)
        training_context = self._detect_training_context(msg_lower)
        escalation = self._check_escalation(msg_lower)
        churn = self._check_churn(msg_lower)
        competitors = self._find_competitors(msg_lower)

        analysis = self._build_analysis(message, emotion, intensity, intent, confidence,
            athlete_type, motivation_level, training_context, escalation, churn, competitors)
        analysis.tokens_used = 0
        analysis.used_ai = False
        return analysis

    def _ai_analyze(self, message: str, history: Optional[List] = None, user_profile: Optional[Dict] = None) -> ContextAnalysis:
        history_context = ""
        if history:
            recent = history[-self._llm["context_history_turns"]:]
            history_context = "\n".join([f"User: {t.user_message}\nAssistant: {t.assistant_response}" for t in recent])

        schema = self._cfg["ai_analysis_schema"]
        system_prompt = f"""You are an expert at analyzing customer messages for a brand.
Analyze the following message and return a JSON object with these fields:
{schema}
{f"Previous conversation context:{history_context}" if history_context else ""}
Return ONLY valid JSON, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=self._llm["context_model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            tokens = response.usage.total_tokens if hasattr(response, "usage") else 0
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = re.sub(r"```(?:json)?", "", content).strip().rstrip("```").strip()
            data = json.loads(content)
        except Exception:
            data = {}
            tokens = 0

        analysis = self._build_from_ai(message, data)
        analysis.tokens_used = tokens
        analysis.used_ai = True
        return analysis

    # ── rule-based helpers ────────────────────────────────────────────────────

    def _detect_emotion(self, msg_lower: str, message: str) -> Tuple[str, int, bool]:
        for emotion, cfg in self._cfg["emotion_keyword_patterns"].items():
            if any(kw in msg_lower for kw in cfg["keywords"]):
                return emotion, cfg["intensity"], True
        return "neutral", 2, False

    def _detect_intent(self, msg_lower: str, message: str, emotion: str) -> Tuple[Intent, float, bool]:
        for pattern in self._cfg["intent_keyword_patterns"]:
            if any(kw in msg_lower for kw in pattern["keywords"]):
                try:
                    intent = Intent(pattern["intent"])
                except ValueError:
                    continue
                return intent, pattern["confidence"], True
        return Intent.PRODUCT_QUESTION, 0.6, False

    def _detect_athlete_type(self, msg_lower: str) -> str:
        kws = self._cfg["athlete_type_keywords"]
        for athlete_type, keywords in kws.items():
            if any(kw in msg_lower for kw in keywords):
                return athlete_type
        return "casual"

    def _detect_motivation_level(self, msg_lower: str, emotion: str, history: Optional[List]) -> str:
        cfg = self._cfg
        if emotion in ("defeated", "frustrated"):
            return "low"
        if emotion in ("motivated", "determined", "excited"):
            return "high"
        if any(w in msg_lower for w in cfg["motivation_keywords"]["low"]):
            return "low"
        if any(w in msg_lower for w in cfg["motivation_keywords"]["high"]):
            return "high"
        if history and len(history) >= 2:
            recent = [getattr(t, "user_emotion", "neutral") for t in history[-2:]]
            if recent.count("defeated") >= 2: return "low"
            if recent.count("motivated") >= 2: return "high"
        return "medium"

    def _detect_training_context(self, msg_lower: str) -> Optional[str]:
        cfg = self._cfg
        for keyword, context in cfg["training_context_keywords"].items():
            if keyword in msg_lower:
                return context
        if any(w in msg_lower for w in cfg["training_context_beginner_words"]):
            return "beginner"
        return None

    def _check_escalation(self, msg_lower: str) -> List[str]:
        return [w for w in self._cfg["escalation_words"] if w in msg_lower]

    def _check_churn(self, msg_lower: str) -> List[str]:
        return [p for p in self._cfg["churn_phrases"] if p in msg_lower]

    def _find_competitors(self, msg_lower: str) -> List[str]:
        return [c for c in self._cfg["competitor_brands"] if c in msg_lower]

    # ── builders ──────────────────────────────────────────────────────────────

    def _build_analysis(self, message, emotion, intensity, intent, confidence,
                         athlete_type, motivation_level, training_context,
                         escalation_signals, churn_signals, competitor_mentions) -> ContextAnalysis:
        cfg = self._cfg
        state_map = cfg["emotion_to_user_state"]
        intent_to_conv = cfg["intent_to_conversation_type"]

        intent_val = intent.value
        conv_type_val = intent_to_conv.get(intent_val, "product_inquiry")
        conv_type = ConversationType(conv_type_val) if conv_type_val in [e.value for e in ConversationType] else ConversationType.PRODUCT_INQUIRY

        urgency = cfg["urgency_rules"]["base_urgency"]
        if intent_val in cfg["urgency_rules"]["high_urgency_intents"] and intensity >= cfg["urgency_rules"]["high_urgency_min_intensity"]:
            urgency = cfg["urgency_rules"]["high_urgency_value"]
        elif intent_val == "order_inquiry":
            urgency = cfg["urgency_rules"]["order_inquiry_urgency"]
        elif any(w in message.lower() for w in cfg["urgency_rules"]["keyword_urgency_words"]):
            urgency = cfg["urgency_rules"]["keyword_urgency_value"]

        emotion_vals = cfg["emotion_sentiment_values"]
        sentiment = emotion_vals.get(emotion, 0.0) * (intensity / 3.0)

        situation_map = cfg["situation_map"]
        if emotion == "injured":
            situation = "injury_setback"
        else:
            situation = situation_map.get(intent_val, "general_inquiry")

        pain_points = []
        msg_lower = message.lower()
        for point, keywords in cfg["pain_point_keywords"].items():
            if any(w in msg_lower for w in keywords):
                pain_points.append(point)

        defaults = cfg["field_defaults"]
        return ContextAnalysis(
            primary_emotion=emotion, emotion_intensity=intensity, sentiment_score=sentiment,
            primary_intent=intent, confidence=confidence, conversation_type=conv_type,
            user_state=state_map.get(emotion, "neutral"), urgency_level=urgency,
            frustration_level=intensity if emotion in ("angry", "frustrated") else 1,
            motivation_level=motivation_level, training_context=training_context,
            athlete_type=athlete_type,
            formality_preference=defaults["formality_preference"],
            product_knowledge=defaults["product_knowledge"],
            technical_level=defaults["technical_level"],
            situation_type=situation, key_pain_points=pain_points,
            time_sensitivity=urgency >= 4, escalation_indicators=escalation_signals,
            churn_indicators=churn_signals, competitor_mentions=competitor_mentions,
            injury_mentioned="injured" in emotion or training_context == "injury_recovery",
        )

    def _build_from_ai(self, message: str, data: Dict) -> ContextAnalysis:
        emotion    = data.get("emotion", "neutral")
        intensity  = int(data.get("emotion_intensity", 2) or 2)
        intent_str = data.get("intent", "product_question")
        confidence = float(data.get("confidence", 0.7) or 0.7)

        training_context = data.get("training_context")
        if training_context in ("null", "none", "", None):
            training_context = None

        try:
            intent = Intent(intent_str)
        except ValueError:
            intent = Intent.PRODUCT_QUESTION

        return self._build_analysis(
            message, emotion, intensity, intent, confidence,
            data.get("athlete_type", "casual"),
            data.get("motivation_level", "medium"),
            training_context,
            data.get("escalation_signals", []) or [],
            data.get("churn_signals", []) or [],
            data.get("competitor_mentions", []) or [],
        )


# Backwards-compatible alias
ApexStrideContextEngine = ContextUnderstandingEngine
