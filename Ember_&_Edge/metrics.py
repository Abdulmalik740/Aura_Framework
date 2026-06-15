"""
metrics.py — Ember & Edge — EMNLP 2026 Industry Track
======================================================
Four rule-based metrics, each bucket-aware and discriminative:

  AC  — Accountability Compliance     (complaint / escalation buckets)
  CCS — Competitor Compliance Score   (competitor bucket)
  KSS — Knife Selection Score         (product_question bucket)
  PDS — Purchase Decisiveness Score   (purchase_intent bucket)
  ECS — Edge Case Compliance Score    (edge_case bucket)
"""

import re
from typing import Dict, List, Optional

_SEVERITY_DEDUCTIONS = {
    "critical": 0.40,
    "high":     0.25,
    "medium":   0.10,
    "low":      0.05,
}

# =============================================================================
# AC: Accountability Compliance
# =============================================================================

_ACCOUNTABILITY_PATTERNS = [
    "that's not right", "thats not right", "that is not right",
    "that shouldn't happen", "that should not happen",
    "that shouldn't have happened", "that's on us", "thats on us",
    "that is on us", "this is on us", "our fault", "we own this",
    "we failed you", "we let you down", "that's our fault",
    "this isn't right", "this is not right", "not what should happen",
]

_CORPORATE_APOLOGIES = [
    "we apologize", "we're sorry", "sorry to hear", 
    "apologize for any inconvenience", "we take this seriously",
    "rest assured", "please allow", "we'll look into",
    "thank you for contacting", "thank you for bringing",
    "we appreciate", "valued customer",
]

_VAGUE_PROMISES = [
    "we'll do our best", "someone will contact", "in touch with you",
    "we'll look into it", "we'll investigate", "team will reach out",
]

_COMPLAINT_REQUIRED_INTENTS = {
    "complaint", "product_issue", "return_refund",
    "replacement", "fear_concern",
}
_COMPLAINT_REQUIRED_BUCKETS = {"complaint", "escalation"}


def accountability_compliance(response: str, intent: str, bucket: str) -> Dict:
    required = (intent in _COMPLAINT_REQUIRED_INTENTS or 
                bucket in _COMPLAINT_REQUIRED_BUCKETS)
    if not required:
        return {"score": 1.0, "scaled": 5.0, "required": False,
                "found": False, "pattern": "n/a"}

    resp_lower = response.lower()
    
    # Must have ownership phrase
    has_ownership = any(pattern in resp_lower for pattern in _ACCOUNTABILITY_PATTERNS)
    
    # Must NOT have corporate apology phrases
    has_corporate = any(apology in resp_lower for apology in _CORPORATE_APOLOGIES)
    
    # Must NOT have vague promises
    has_vague = any(vp in resp_lower for vp in _VAGUE_PROMISES)
    
    # Check for concrete fix (replacement, refund, send, etc.)
    has_concrete_fix = any(fix in resp_lower for fix in 
                          ["replace", "refund", "send", "ship", "new one", "fix", "sorted"])
    
    # Score calculation
    if has_ownership and not has_corporate and not has_vague and has_concrete_fix:
        score = 1.0  # Perfect
    elif has_ownership and not has_corporate and not has_vague:
        score = 0.8  # Ownership but no concrete fix mentioned
    elif has_ownership and (has_corporate or has_vague):
        score = 0.5  # Mixed - has ownership but corporate fluff
    elif not has_ownership and not has_corporate:
        score = 0.2  # Missed ownership but at least not corporate
    else:
        score = 0.0  # Corporate apology with no ownership
    
    scaled = round(score * 5, 2)
    
    return {"score": score, "scaled": scaled, "required": True,
            "found": has_ownership, "pattern": "found" if has_ownership else "",
            "has_corporate": has_corporate, "has_concrete_fix": has_concrete_fix}


# =============================================================================
# CCS: Competitor Compliance Score
# =============================================================================

_COMPETITOR_NAMES = [
    "wusthof", "wüsthof", "shun", "global", "victorinox",
    "henckels", "miyabi", "zwilling", "mac knife", "misono",
    "tojiro", "kai", "dalstrong", "misen",
]

_REFRAME_PIVOT_PATTERNS = [
    " but our", " but we", " however", " however,", ", but ",
    " our knives", " our artisan", " our approach", " we focus",
    " the difference", " what sets", " instead,",
    " yet,", " yet ", "yet,", "yet ", " what matters", " what truly",
    " ours ", " we believe", " for us,", ". at ember", " we offer",
]

_CRAFT_PHILOSOPHY_PATTERNS = [
    "serves the ingredient", "knife serves", "the ingredient",
    "feel before technique", "every cut", "forged, not", "forged not",
    "craft", "presence", "sensory", "what you're cutting",
    "what are you cutting", "the blade whispers", "whispers through",
    "weight and balance", "feel the",
    "art of cooking", "art of", "reconnect", "reconnecting",
    "connection", "intention", "glides", "effortlessly",
    "moment of", "inspire", "cooking experience", "culinary", "ingredients",
]

_FEATURE_COMPARE_WORDS = [
    "better specs", "higher hrc", "more advanced", "superior technology",
    "better than", "outperform", "more cushioning", "compared to",
    "better edge", "sharper than", "harder steel", "better quality than",
    "superior blade", "surpasses", "beats", "hrc", "rockwell",
]


def competitor_compliance_score(response: str, bucket: str) -> Dict:
    if bucket != "competitor":
        return {"score": 1.0, "scaled": 5.0, "required": False,
                "components": {}, "detail": "n/a"}

    resp_lower = response.lower()
    philosophy = any(p in resp_lower for p in _CRAFT_PHILOSOPHY_PATTERNS)
    no_feature = not any(f in resp_lower for f in _FEATURE_COMPARE_WORDS)

    comp_names_found = [c for c in _COMPETITOR_NAMES if c in resp_lower]
    if comp_names_found:
        acknowledgment_only = any(pivot in resp_lower for pivot in _REFRAME_PIVOT_PATTERNS)
        if acknowledgment_only and no_feature:
            no_comp_score = 0.8
            comp_status = "acknowledged_reframe"
        else:
            no_comp_score = 0.0
            comp_status = "direct_comparison"
    else:
        no_comp_score = 1.0
        comp_status = "not_mentioned"

    score = (int(philosophy) * 0.4 + no_comp_score * 0.4 + int(no_feature) * 0.2)
    scaled = round(score * 5, 2)

    return {
        "score": round(score, 3), "scaled": scaled, "required": True,
        "components": {
            "philosophy_present": philosophy,
            "competitor_status": comp_status,
            "no_feature_compare": no_feature,
        },
        "hits": {
            "competitor_names": comp_names_found,
            "feature_words": [f for f in _FEATURE_COMPARE_WORDS if f in resp_lower],
            "philosophy_found": [p for p in _CRAFT_PHILOSOPHY_PATTERNS if p in resp_lower],
        }
    }



# =============================================================================
# KSS: Knife Selection Score
# =============================================================================

_VALID_KNIVES = [
    "artisan chef's knife", "artisan chef knife",
    "artisan chef", "paring knife", "bread knife",
    "santoku", "beginner set", "chef's knife", "chef knife",
]

_QUESTION_OR_GUIDE_SIGNALS = [
    "what do you", "what are you", "what do you find", "tell me",
    "where do you", "how often", "what kind of", "do you mostly",
    "what draws you", "?", "let me ask", "start there",
    "begin with", "ready to start", "ready when you are",
]

_LISTING_PHRASES = [
    "also consider", "another option", "alternatively", "other options",
    "you could also try", "both the", "either the", "or the", "as well as",
    "we also offer", "we have several", "first", "second", "third", "1.", "2.", "3.",
]

_TECHNICAL_SPECS = ["hrc", "rockwell", "degrees", "angle", "steel composition", "geometry"]


def knife_selection_score(response: str, bucket: str) -> Dict:
    if bucket != "product_question":
        return {"score": 1.0, "scaled": 5.0, "required": False,
                "components": {}, "word_count": len(response.split())}

    resp_lower = response.lower()
    words = resp_lower.split()
    first_15 = " ".join(words[:15])

    knife_first = any(k in first_15 for k in _VALID_KNIVES)

    unique_knives = []
    for k in _VALID_KNIVES:
        if k in resp_lower and k not in unique_knives:
            unique_knives.append(k)

    has_listing = any(phrase in resp_lower for phrase in _LISTING_PHRASES)
    single_knife = len(unique_knives) == 1 and not has_listing

    has_question = any(sig in resp_lower for sig in _QUESTION_OR_GUIDE_SIGNALS)

    # Short decisive responses (under 35 words) with knife named count as having guidance
    if len(words) <= 35 and knife_first:
        has_question = True
    
    # Penalty for technical specs
    has_tech_specs = any(spec in resp_lower for spec in _TECHNICAL_SPECS)
    tech_penalty = 0.3 if has_tech_specs else 0

    score = (int(knife_first) + int(single_knife) + int(has_question)) / 3.0
    score = max(0.0, score - tech_penalty)
    scaled = round(score * 5, 2)

    return {
        "score": round(score, 3), "scaled": scaled, "required": True,
        "components": {
            "knife_named_first": knife_first,
            "single_knife": single_knife,
            "question_or_guide": has_question,
        },
        "knives_found": unique_knives,
        "listing_hits": [p for p in _LISTING_PHRASES if p in resp_lower],
        "technical_specs_found": has_tech_specs,
        "word_count": len(words),
    }


# =============================================================================
# PDS: Purchase Decisiveness Score
# =============================================================================

_PUSHY_LANGUAGE = [
    "buy now", "don't miss out", "limited time", "act now",
    "hurry", "selling fast", "order now before", "while stocks last",
    "exclusive offer", "special deal",
]

_HEDGING_PHRASES = [
    "i can't check", "i cannot check", "i'm unable to check",
    "i don't have access", "i can't confirm", "cannot confirm",
    "you may want to", "i'd suggest checking",
    "i encourage you to visit", "reach out to", "contact our",
    "i recommend visiting", "please check", "you'll need to",
]

_POETIC_FILLER_ON_PURCHASE = [
    "culinary journey", "culinary creation", "culinary adventure",
    "wonderful tool", "beautiful tool", "truly serves",
    "transform your", "enhance your experience", "elevate your",
    "explore its feel", "explore how it feels",
]


def purchase_decisiveness_score(response: str, bucket: str) -> Dict:
    """PDS: discriminates across conditions — measures decisiveness on transactional questions."""
    if bucket != "purchase_intent":
        return {"score": 1.0, "scaled": 5.0, "required": False,
                "components": {}, "word_count": len(response.split())}

    resp_lower = response.lower()
    words = resp_lower.split()
    word_count = len(words)

    first_10 = " ".join(words[:10])
    immediate_answer = any(sig in first_10 for sig in ["yes", "in stock", "available", "no,", "not in stock"])
    no_hedging = not any(h in resp_lower for h in _HEDGING_PHRASES)
    word_economy = word_count <= 40
    no_poetic_filler = not any(p in resp_lower for p in _POETIC_FILLER_ON_PURCHASE)
    no_pushy = not any(p in resp_lower for p in _PUSHY_LANGUAGE)

    score = (int(immediate_answer) + int(no_hedging) + int(word_economy) + int(no_poetic_filler)) / 4.0
    if not no_pushy:
        score = 0.0
    scaled = round(score * 5, 2)

    return {
        "score": round(score, 3), "scaled": scaled, "required": True,
        "components": {
            "immediate_answer": immediate_answer,
            "no_hedging": no_hedging,
            "word_economy": word_economy,
            "no_poetic_filler": no_poetic_filler,
            "no_pushy_language": no_pushy,
        },
        "word_count": word_count,
        "hits": {
            "hedging": [h for h in _HEDGING_PHRASES if h in resp_lower],
            "poetic_filler": [p for p in _POETIC_FILLER_ON_PURCHASE if p in resp_lower],
            "pushy": [p for p in _PUSHY_LANGUAGE if p in resp_lower],
        },
    }

# =============================================================================
# ECS: Edge Case Compliance Score
# =============================================================================

_EDGE_REQUIRED_PATTERNS = [
    "that's not right", "thats not right", "that is not right",
    "that shouldn't happen", "that should not happen",
    "this is on us", "our fault", "we failed you",
]

_EDGE_FORBIDDEN_PATTERNS = [
    "we apologize for any inconvenience", "sorry to hear that",
    "thank you for contacting", "we understand your frustration",
    "we take this very seriously", "rest assured",
    "please allow", "we'll do our best", "someone will be in touch",
    "is there anything else", "feel free to", "let me know if",
]

_EDGE_INJURY_HANDLING = [
    "heal", "finger", "care for", "take care", "safe", "okay",
    "hope you're", "feel better",
]

_EDGE_POETRY_WORDS = ["whisper", "glide", "balance", "sensory", "poetic", "presence", "rhythm"]


def edge_case_compliance(response: str, bucket: str, intent: str, original_message: str = "") -> Dict:
    """Edge cases require: ownership + no corporate + injury awareness (if applicable)"""
    if bucket != "edge_case":
        return {"score": 1.0, "scaled": 5.0, "required": False,
                "components": {}}

    resp_lower = response.lower()
    orig_lower = original_message.lower()
    
    # Must take ownership of product failure
    has_ownership = any(p in resp_lower for p in _EDGE_REQUIRED_PATTERNS)
    
    # Must NOT use corporate/generic apology language
    no_corporate = not any(p in resp_lower for p in _EDGE_FORBIDDEN_PATTERNS)
    
    # Check if injury was mentioned in original message
    injury_keywords = ["cut", "slice", "bleed", "hurt", "injury", "finger", "hand"]
    injury_mentioned = any(ik in orig_lower for ik in injury_keywords)
    
    # If injury mentioned, must acknowledge it appropriately
    injury_handled = True
    if injury_mentioned:
        injury_handled = any(ih in resp_lower for ih in _EDGE_INJURY_HANDLING)
    
    # Penalty for mentioning competitors in edge cases
    competitor_penalty = 0
    for comp in _COMPETITOR_NAMES:
        if comp in resp_lower:
            competitor_penalty = 0.3
            break
    
    # Penalty for poetry/sensory language in edge cases (should be direct)
    sensory_penalty = 0
    for sw in _EDGE_POETRY_WORDS:
        if sw in resp_lower:
            sensory_penalty = 0.2
            break
    
    # Check for concrete action
    has_action = any(action in resp_lower for action in 
                    ["send", "replace", "refund", "fix", "help", "address", "photo"])
    
    base_score = (int(has_ownership) + int(no_corporate) + int(has_action)) / 3.0
    
    if injury_mentioned:
        base_score = (base_score + int(injury_handled)) / 2.0
    
    final_score = max(0.0, min(1.0, base_score - competitor_penalty - sensory_penalty))
    scaled = round(final_score * 5, 2)
    
    return {
        "score": round(final_score, 3), "scaled": scaled, "required": True,
        "components": {
            "has_ownership": has_ownership,
            "no_corporate_language": no_corporate,
            "has_concrete_action": has_action,
            "injury_handled": injury_handled if injury_mentioned else "n/a",
        },
        "penalties": {
            "competitor_applied": competitor_penalty > 0,
            "sensory_applied": sensory_penalty > 0,
        },
        "injury_mentioned": injury_mentioned,
    }


# =============================================================================
# Dispatcher and Helpers
# =============================================================================

_BUCKET_METRIC_MAP = {
    "complaint": "ac", "escalation": "ac",
    "competitor": "ccs", "product_question": "kss",
    "purchase_intent": "pds", "edge_case": "ecs",
}


def compute_third_metric(response: str, bucket: str, intent: str, expected_contains: List[str], original_message: str = "") -> Dict:
    metric_name = _BUCKET_METRIC_MAP.get(bucket, None)

    if metric_name == "ac":
        r = accountability_compliance(response, intent, bucket)
        return {"name": "AC", "score": r["score"], "scaled": r["scaled"], "detail": r}
    elif metric_name == "ccs":
        r = competitor_compliance_score(response, bucket)
        return {"name": "CCS", "score": r["score"], "scaled": r["scaled"], "detail": r}
    elif metric_name == "kss":
        r = knife_selection_score(response, bucket)
        return {"name": "KSS", "score": r["score"], "scaled": r["scaled"], "detail": r}
    elif metric_name == "pds":
        r = purchase_decisiveness_score(response, bucket)
        return {"name": "PDS", "score": r["score"], "scaled": r["scaled"], "detail": r}
    elif metric_name == "ecs":
        r = edge_case_compliance(response, bucket, intent, original_message)
        return {"name": "ECS", "score": r["score"], "scaled": r["scaled"], "detail": r}
    else:
        return {"name": "N/A", "score": 1.0, "scaled": 5.0, "detail": {}}


def severity_weighted_guard_score(violations: List[Dict]) -> Dict:
    total_deducted = 0.0
    by_severity: Dict[str, int] = {}
    by_category: Dict[str, int] = {}

    for v in violations:
        sev = v.get("severity", "medium")
        cat = v.get("category", "unknown")
        deduction = _SEVERITY_DEDUCTIONS.get(sev, 0.10)
        total_deducted += deduction
        by_severity[sev] = by_severity.get(sev, 0) + 1
        by_category[cat] = by_category.get(cat, 0) + 1

    score = max(0.0, round(1.0 - total_deducted, 3))
    return {
        "score": score, "total_deducted": round(total_deducted, 3),
        "by_severity": by_severity, "by_category": by_category,
        "violation_count": len(violations),
    }


def compute_all_metrics(response: str, intent: str, bucket: str,
                        expected_contains: Optional[List[str]] = None,
                        violations: Optional[List[Dict]] = None,
                        original_message: str = "") -> Dict:
    expected_contains = expected_contains or []
    violations = violations or []

    third = compute_third_metric(response, bucket, intent, expected_contains, original_message)
    swg = severity_weighted_guard_score(violations)

    return {
        "third_metric_name": third["name"],
        "third_metric_score": third["scaled"],
        "third_metric_detail": third["detail"],
        "swg": round(swg["score"] * 5, 2),
        "swg_detail": swg,
        "word_count": len(response.split()),
    }


def violation_breakdown_report(all_results: List[Dict]) -> Dict:
    total_responses = len(all_results)
    by_category: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    by_condition: Dict[str, Dict] = {}
    by_bucket: Dict[str, Dict] = {}
    clean_count = 0

    for r in all_results:
        viols = r.get("brand_validation", {}).get("violations", [])
        cond = r.get("condition", "?")
        buck = r.get("bucket", "?")

        if not viols:
            clean_count += 1

        if cond not in by_condition:
            by_condition[cond] = {"total_violations": 0, "responses": 0, "by_category": {}}
        by_condition[cond]["responses"] += 1

        if buck not in by_bucket:
            by_bucket[buck] = {"total_violations": 0, "responses": 0}
        by_bucket[buck]["responses"] += 1

        for v in viols:
            cat = v.get("category", "unknown")
            sev = v.get("severity", "medium")
            by_category[cat] = by_category.get(cat, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1
            by_condition[cond]["total_violations"] += 1
            by_condition[cond]["by_category"][cat] = by_condition[cond]["by_category"].get(cat, 0) + 1
            by_bucket[buck]["total_violations"] += 1

    by_category_sorted = dict(sorted(by_category.items(), key=lambda x: x[1], reverse=True))

    return {
        "total_responses": total_responses, "clean_responses": clean_count,
        "clean_rate": round(clean_count / max(total_responses, 1), 3),
        "by_category": by_category_sorted, "by_severity": by_severity,
        "by_condition": by_condition, "by_bucket": by_bucket,
        "top_3_categories": list(by_category_sorted.keys())[:3],
    }


def print_violation_report(report: Dict):
    print(f"\n{'='*60}")
    print("VIOLATION BREAKDOWN REPORT")
    print(f"{'='*60}")
    print(f"Total responses : {report['total_responses']}")
    print(f"Clean responses : {report['clean_responses']} ({report['clean_rate']:.0%})")
    print(f"\n{'─'*60}")
    print("BY CATEGORY (frequency):")
    for cat, count in report["by_category"].items():
        bar = "█" * min(count, 40)
        print(f"  {cat:<35} {count:3d}  {bar}")
    print(f"\n{'─'*60}")
    print("BY SEVERITY:")
    for sev in ["critical", "high", "medium", "low"]:
        count = report["by_severity"].get(sev, 0)
        print(f"  {sev:<10} {count:3d}")
    print(f"\n{'─'*60}")
    print("BY CONDITION (total violations):")
    for cond in sorted(report["by_condition"].keys()):
        d = report["by_condition"][cond]
        rate = d["total_violations"] / max(d["responses"], 1)
        print(f"  Condition {cond}: {d['total_violations']:3d} violations across {d['responses']} responses ({rate:.1f}/response)")
    print(f"{'='*60}\n")