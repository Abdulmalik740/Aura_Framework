"""
Brand Consistency Guard
All forbidden patterns, alternatives, thresholds, and context rules
loaded from brand_config.yaml.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import re
from openai import OpenAI
from config_loader import brand_guard_config


@dataclass
class BrandViolation:
    severity: str
    category: str
    description: str
    suggestion: Optional[str] = None


@dataclass
class BrandValidationResult:
    overall_score: float
    passed: bool
    violations: List[BrandViolation]
    detailed_feedback: str
    scores: Dict[str, float] = field(default_factory=dict)


class BrandConsistencyGuard:
    """
    Brand guard driven entirely by brand_config.yaml.
    No forbidden phrases, thresholds, or rules hardcoded here.
    """

    def __init__(self, brand_config: Dict, client: OpenAI, threshold: float = None):
        self.brand_config = brand_config
        self.client = client
        self._cfg = brand_guard_config()
        self.threshold = threshold if threshold is not None else self._cfg["validation_threshold"]
        self._forbidden = self._cfg["forbidden_patterns"]
        self._alternatives = self._cfg["forbidden_alternatives"]
        self._dna = self._cfg["linguistic_dna"]
        self._weights = self._cfg["scoring_weights"]
        self._context_rules = self._cfg["context_rules"]

    def validate(
        self,
        response: str,
        context: Optional[Dict] = None,
        conversation_type: Optional[str] = None,
    ) -> BrandValidationResult:
        violations = []
        scores = {}

        forbidden_score, forbidden_viols = self._check_forbidden(response)
        violations.extend(forbidden_viols)
        scores["forbidden_check"] = forbidden_score

        dna_score, dna_viols = self._check_linguistic_dna(response)
        violations.extend(dna_viols)
        scores["linguistic_dna"] = dna_score

        context_score, context_viols = self._check_context_tone(response, conversation_type, context)
        violations.extend(context_viols)
        scores["context_appropriateness"] = context_score

        overall_score = sum(scores[k] * self._weights[k] for k in scores) / sum(self._weights.values())
        has_critical = any(v.severity == "critical" for v in violations)
        passed = overall_score >= self.threshold and not has_critical

        feedback = self._generate_feedback(overall_score, scores, violations)
        return BrandValidationResult(overall_score=overall_score, passed=passed, violations=violations, detailed_feedback=feedback, scores=scores)

    def _check_forbidden(self, response: str) -> Tuple[float, List[BrandViolation]]:
        violations = []
        response_lower = response.lower()
        deduction = self._cfg["violation_score_deduction"]

        for category, cat_cfg in self._forbidden.items():
            severity = cat_cfg["severity"]
            for phrase in cat_cfg["phrases"]:
                if phrase in response_lower:
                    violations.append(BrandViolation(
                        severity=severity,
                        category=f"forbidden_{category}",
                        description=f"Forbidden phrase: '{phrase}'",
                        suggestion=self._alternatives.get(phrase, "Rewrite: short, direct, action-focused"),
                    ))

        score = max(0.0, 1.0 - len(violations) * deduction)
        return score, violations

    def _check_linguistic_dna(self, response: str) -> Tuple[float, List[BrandViolation]]:
        violations = []
        dna = self._dna

        sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
        if not sentences:
            return 0.5, violations

        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_length > dna["max_avg_sentence_length"]:
            violations.append(BrandViolation(
                severity="medium",
                category="linguistic_dna",
                description=f"Sentences too long ({avg_length:.0f} words avg). Target: {dna['max_avg_sentence_length']} words.",
                suggestion="Break into shorter sentences. 'Lace up. Move out.' Not paragraphs.",
            ))

        contractions = len(re.findall(
            r"\b(you're|it's|that's|don't|can't|won't|we're|I'm|hasn't|wouldn't)\b",
            response, re.IGNORECASE,
        ))
        words = len(response.split())
        contraction_ratio = contractions / max(words, 1) if words > dna["min_word_count_for_ratio"] else 1.0

        if contraction_ratio < dna["min_contraction_ratio"] and words > dna["min_word_count_for_ratio"]:
            violations.append(BrandViolation(
                severity="medium",
                category="linguistic_dna",
                description=f"Low contraction usage ({contraction_ratio:.0%}). Sounds too formal.",
                suggestion="Use contractions: 'you're' not 'you are', 'don't' not 'do not'",
            ))

        score = 1.0 if not violations else 0.6
        return score, violations

    def _check_context_tone(self, response: str, conversation_type: Optional[str], context: Optional[Dict]) -> Tuple[float, List[BrandViolation]]:
        violations = []
        deduction = self._cfg["context_score_deduction"]
        response_lower = response.lower()

        if not conversation_type:
            return 1.0, violations

        rules = self._context_rules

        if conversation_type == "complaint_resolution":
            r = rules["complaint_resolution"]
            if not any(w in response_lower for w in r["required_acknowledgment_words"]):
                violations.append(BrandViolation(severity="critical", category="complaint_tone",
                    description="Missing accountability in complaint response",
                    suggestion="Own it: 'That's on us.' 'Frustrating, we know.'"))
            if not any(w in response_lower for w in r["required_solution_words"]):
                violations.append(BrandViolation(severity="critical", category="complaint_tone",
                    description="Missing concrete solution in complaint response",
                    suggestion="State the fix: 'Replacement ships today.' 'Full refund processed.'"))
            if r["forbidden_generic_apology"] in response_lower:
                violations.append(BrandViolation(severity="high", category="complaint_tone",
                    description="Generic corporate apology",
                    suggestion="Be direct: 'That's on us. Here's the fix.' Not generic 'we apologize'."))

        elif conversation_type == "product_comparison":
            r = rules["product_comparison"]
            if not any(p in response_lower for p in r["required_philosophy_phrases"]):
                violations.append(BrandViolation(severity="critical", category="comparison_tone",
                    description="Missing 'You vs. You' philosophy in competitor comparison",
                    suggestion="Include core message: 'You vs. You. That's the difference.'"))
            if any(w in response_lower for w in r["forbidden_feature_comparison_words"]):
                violations.append(BrandViolation(severity="high", category="comparison_tone",
                    description="Competing on features instead of philosophy",
                    suggestion=r["philosophy_suggestion"]))

        elif conversation_type == "purchase_intent":
            r = rules["purchase_intent"]
            if len(response.split()) > r["max_word_count"]:
                violations.append(BrandViolation(severity="medium", category="purchase_tone",
                    description="Purchase response too long. Be decisive.",
                    suggestion=f"Under {r['max_word_count']} words. State facts. 'In stock. Ships today. Your move.'"))
            if any(w in response_lower for w in r["forbidden_pushy_words"]):
                violations.append(BrandViolation(severity="critical", category="purchase_tone",
                    description="Pushy sales language",
                    suggestion="Be confident, not desperate. State facts only."))

        # General checks
        gen = rules["general"]
        if any(w in response_lower for w in gen["forbidden_cheesy_motivation"]):
            violations.append(BrandViolation(severity="high", category="tone_boundary",
                description="Cheesy motivation language",
                suggestion="Be direct: 'Prove it.' 'Show up.' 'Make it count.'"))

        if conversation_type not in gen.get("soft_language_exempt_types", []):
            if any(w in response_lower for w in gen["forbidden_soft_language"]):
                violations.append(BrandViolation(severity="medium", category="tone_boundary",
                    description="Soft/coddling language",
                    suggestion="Challenge instead: 'Time to move.' 'Start now.'"))

        score = max(0.0, 1.0 - len(violations) * deduction)
        return score, violations

    def _generate_feedback(self, overall_score: float, scores: Dict[str, float], violations: List[BrandViolation]) -> str:
        lines = [f"Brand Alignment: {overall_score:.0%}\n"]
        for category, score in scores.items():
            status = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
            lines.append(f"{status} {category.replace('_', ' ').title()}: {score:.0%}")
        if violations:
            lines.append(f"\n🚨 {len(violations)} issue(s):")
            for v in violations[:3]:
                lines.append(f"  • {v.description}")
                if v.suggestion:
                    lines.append(f"    → {v.suggestion}")
        else:
            lines.append("\n✅ Clean brand voice!")
        return "\n".join(lines)


# Backwards-compatible alias
ApexStrideBrandGuard = BrandConsistencyGuard
