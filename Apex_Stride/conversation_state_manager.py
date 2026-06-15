"""
Conversation State Manager
All thresholds, keyword lists, and mappings loaded from brand_config.yaml.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from config_loader import conversation_state_config


class ConversationPhase(Enum):
    INITIAL_CONTACT = "initial_contact"
    ISSUE_ASSESSMENT = "issue_assessment"
    SOLUTION_DELIVERY = "solution_delivery"
    MOTIVATION_BOOST = "motivation_boost"
    PURCHASE_INTENT = "purchase_intent"
    RESOLUTION = "resolution"
    FOLLOW_UP = "follow_up"
    CLOSING = "closing"


class MotivationTrend(Enum):
    RISING = "rising"
    DECLINING = "declining"
    STABLE_HIGH = "stable_high"
    STABLE_LOW = "stable_low"
    RECOVERING = "recovering"
    VOLATILE = "volatile"


@dataclass
class ConversationTurn:
    timestamp: datetime
    user_message: str
    assistant_response: str
    user_emotion: str
    emotion_intensity: int
    intent: str
    topic: str
    motivation_level: str
    sentiment_score: float
    urgency_level: int
    tone_used: Dict
    context_snapshot: Dict


@dataclass
class AthleteProfile:
    user_id: Optional[str] = None
    athlete_type: str = "casual"
    motivation_baseline: str = "medium"
    training_context: Optional[str] = None
    injury_history: List[str] = field(default_factory=list)
    product_knowledge: str = "new_to_brand"
    owned_products: List[str] = field(default_factory=list)
    favorite_product: Optional[str] = None
    formality_preference: str = "casual"
    communication_style: str = "direct"
    preferred_response_length: str = "brief"
    inferred_age_group: Optional[str] = None
    age_confidence: str = "low"
    relationship_stage: str = "first_interaction"
    total_interactions: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    churn_risk_score: float = 0.0
    competitor_mentions: List[str] = field(default_factory=list)
    considering_switch: bool = False

    def update_from_turn(self, turn: ConversationTurn):
        cfg = conversation_state_config()
        stages = cfg["relationship_stages"]
        self.total_interactions += 1

        if turn.sentiment_score > 0.3:
            self.positive_interactions += 1
        elif turn.sentiment_score < -0.3:
            self.negative_interactions += 1

        if self.total_interactions == 1:
            self.relationship_stage = "first_interaction"
        elif self.total_interactions < stages["building"]:
            self.relationship_stage = "building"
        elif self.positive_interactions / self.total_interactions > stages["loyal_positive_ratio"]:
            self.relationship_stage = "loyal"
        elif self.negative_interactions / self.total_interactions > stages["at_risk_negative_ratio"]:
            self.relationship_stage = "at_risk"

        if turn.motivation_level == "high":
            self.motivation_baseline = "high"
        elif turn.motivation_level == "low" and self.motivation_baseline != "high":
            self.motivation_baseline = "low"

        if turn.context_snapshot.get("athlete_type"):
            self.athlete_type = turn.context_snapshot["athlete_type"]
        if turn.context_snapshot.get("training_context"):
            self.training_context = turn.context_snapshot["training_context"]
        if turn.context_snapshot.get("injury_mentioned"):
            note = f"{turn.timestamp.strftime('%Y-%m-%d')}: mentioned in conversation"
            if note not in self.injury_history:
                self.injury_history.append(note)


@dataclass
class Issue:
    id: str
    description: str
    category: str
    severity: str
    status: str
    first_mentioned: datetime
    last_mentioned: Optional[datetime] = None
    resolution_attempts: int = 0
    resolved: bool = False
    resolution_summary: Optional[str] = None
    affects_training: bool = False
    competitor_comparison: bool = False


class ConversationStateManager:
    """
    Stateful conversation manager.
    All thresholds and keywords come from brand_config.yaml.
    """

    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self._cfg = conversation_state_config()
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.turns: List[ConversationTurn] = []
        self.athlete_profile = AthleteProfile(user_id=user_id)
        self.current_phase = ConversationPhase.INITIAL_CONTACT
        self.topic_stack: List[str] = []
        self.topic_history: List[Tuple[str, datetime]] = []
        self.active_issues: List[Issue] = []
        self.resolved_issues: List[Issue] = []
        self.motivation_history: List[Tuple[datetime, str]] = []
        self.motivation_trend = MotivationTrend.STABLE_HIGH
        self.sentiment_history: List[Tuple[datetime, float]] = []
        self.escalation_risk: float = 0.0
        self.churn_risk: float = 0.0
        self.competitor_switch_risk: float = 0.0
        self.frustration_counter: int = 0
        self.training_goals: List[str] = []
        self.mentioned_products: List[str] = []

    def add_turn(self, user_message: str, assistant_response: str, context: Dict, tone: Dict):
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_message=user_message,
            assistant_response=assistant_response,
            user_emotion=context.get("emotion", "neutral"),
            emotion_intensity=context.get("emotion_intensity", 2),
            intent=context.get("intent", ""),
            topic=context.get("conversation_type", ""),
            motivation_level=context.get("motivation_level", "medium"),
            sentiment_score=context.get("sentiment_score", 0.0),
            urgency_level=context.get("urgency_level", 2),
            tone_used=tone,
            context_snapshot=context,
        )
        self.turns.append(turn)
        self.athlete_profile.update_from_turn(turn)
        self.motivation_history.append((turn.timestamp, turn.motivation_level))
        self.sentiment_history.append((turn.timestamp, turn.sentiment_score))
        self._update_risks(turn, context)
        self._update_phase(turn, context)
        self.last_updated = datetime.now()

    def _update_risks(self, turn: ConversationTurn, context: Dict):
        escalation = context.get("escalation_indicators", [])
        churn = context.get("churn_indicators", [])
        competitors = context.get("competitor_mentions", [])

        if escalation:
            self.escalation_risk = min(1.0, self.escalation_risk + 0.3)
        if churn:
            self.churn_risk = min(1.0, self.churn_risk + 0.3)
        if competitors:
            self.competitor_switch_risk = min(1.0, self.competitor_switch_risk + 0.2)
            self.athlete_profile.competitor_mentions.extend(competitors)

        if turn.sentiment_score < -0.5:
            self.frustration_counter += 1
        elif turn.sentiment_score > 0.3:
            self.frustration_counter = max(0, self.frustration_counter - 1)

        self._update_motivation_trend()

    def _update_motivation_trend(self):
        if len(self.motivation_history) < 2:
            return
        recent = [m[1] for m in self.motivation_history[-3:]]
        if all(m == "high" for m in recent): self.motivation_trend = MotivationTrend.STABLE_HIGH
        elif all(m == "low" for m in recent): self.motivation_trend = MotivationTrend.STABLE_LOW
        elif recent[-1] == "high" and recent[0] == "low": self.motivation_trend = MotivationTrend.RISING
        elif recent[-1] == "low" and recent[0] == "high": self.motivation_trend = MotivationTrend.DECLINING

    def _update_phase(self, turn: ConversationTurn, context: Dict):
        intent = turn.intent
        if intent in ("return_refund", "replacement", "product_issue", "complaint"):
            self.current_phase = ConversationPhase.ISSUE_ASSESSMENT
        elif intent == "motivation_seeking":
            self.current_phase = ConversationPhase.MOTIVATION_BOOST
        elif self._detect_purchase_intent(turn):
            self.current_phase = ConversationPhase.PURCHASE_INTENT
        elif intent == "closing":
            self.current_phase = ConversationPhase.CLOSING

    def _detect_purchase_intent(self, turn: ConversationTurn) -> bool:
        signals = self._cfg["purchase_signals"]
        return any(s in turn.user_message.lower() for s in signals)

    def should_escalate(self) -> bool:
        cfg = self._cfg
        return (
            self.escalation_risk > cfg["escalation_risk_threshold"]
            or self.frustration_counter >= cfg["frustration_counter_threshold"]
            or any(i.severity == "critical" for i in self.active_issues)
            or (self.competitor_switch_risk > cfg["competitor_switch_risk_threshold"]
                and self.churn_risk > cfg["churn_risk_combined_threshold"])
        )

    def needs_motivation_boost(self) -> bool:
        if self.motivation_trend in (MotivationTrend.DECLINING, MotivationTrend.STABLE_LOW):
            return True
        if self.athlete_profile.motivation_baseline == "low":
            return True
        if self.turns and self.turns[-1].user_emotion == "defeated":
            return True
        return self.is_motivation_question()

    def is_motivation_question(self) -> bool:
        if not self.turns:
            return False
        last = self.turns[-1].user_message.lower()
        return any(kw in last for kw in self._cfg["motivation_keywords"])

    def get_conversation_type_for_guard(self) -> str:
        if not self.turns:
            return "general_chat"
        intent = self.turns[-1].intent
        return self._cfg["intent_to_brand_guard_type"].get(intent, "general_chat")

    def get_context_summary(self) -> Dict:
        recent_emotions = [t.user_emotion for t in self.turns[-3:]] if len(self.turns) >= 3 else []
        recent_motivation = [t.motivation_level for t in self.turns[-3:]] if len(self.turns) >= 3 else []
        return {
            "session_id": self.session_id,
            "turn_count": len(self.turns),
            "current_phase": self.current_phase.value,
            "motivation_trend": self.motivation_trend.value,
            "current_motivation": self.motivation_history[-1][1] if self.motivation_history else "medium",
            "recent_emotions": recent_emotions,
            "recent_motivation": recent_motivation,
            "active_topics": self.topic_stack,
            "athlete_profile": {
                "athlete_type": self.athlete_profile.athlete_type,
                "motivation_baseline": self.athlete_profile.motivation_baseline,
                "training_context": self.athlete_profile.training_context,
                "product_knowledge": self.athlete_profile.product_knowledge,
                "relationship_stage": self.athlete_profile.relationship_stage,
                "injury_history": len(self.athlete_profile.injury_history) > 0,
                "considering_switch": self.athlete_profile.considering_switch,
            },
            "training_goals": self.training_goals,
            "active_issues_count": len(self.active_issues),
            "resolved_issues_count": len(self.resolved_issues),
            "risk_flags": {
                "escalation_risk": self.escalation_risk,
                "churn_risk": self.churn_risk,
                "competitor_switch_risk": self.competitor_switch_risk,
                "frustration_counter": self.frustration_counter,
            },
            "competitor_mentions": self.athlete_profile.competitor_mentions,
        }

    def get_conversation_history_text(self, max_turns: int = 5) -> str:
        parts = []
        for i, turn in enumerate(self.turns[-max_turns:], 1):
            parts += [f"Turn {i}:", f"Athlete: {turn.user_message}",
                      f"Response: {turn.assistant_response}", f"Motivation: {turn.motivation_level}", ""]
        return "\n".join(parts)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "turn_count": len(self.turns),
            "current_phase": self.current_phase.value,
            "motivation_trend": self.motivation_trend.value,
            "athlete_profile": {
                "athlete_type": self.athlete_profile.athlete_type,
                "motivation_baseline": self.athlete_profile.motivation_baseline,
                "training_context": self.athlete_profile.training_context,
                "product_knowledge": self.athlete_profile.product_knowledge,
                "relationship_stage": self.athlete_profile.relationship_stage,
                "considering_switch": self.athlete_profile.considering_switch,
                "competitor_mentions": self.athlete_profile.competitor_mentions,
                "churn_risk_score": self.athlete_profile.churn_risk_score,
            },
            "training_goals": self.training_goals,
            "active_issues": [
                {"id": i.id, "description": i.description, "category": i.category,
                 "severity": i.severity, "status": i.status,
                 "affects_training": i.affects_training, "competitor_comparison": i.competitor_comparison}
                for i in self.active_issues
            ],
            "risk_scores": {
                "escalation_risk": self.escalation_risk,
                "churn_risk": self.churn_risk,
                "competitor_switch_risk": self.competitor_switch_risk,
            },
        }


# Backwards-compatible alias
ApexStrideConversationState = ConversationStateManager
