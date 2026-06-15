import re
"""
metrics.py — EMNLP 2026 Industry Track
=======================================
Three rule-based metrics, each bucket-aware and discriminative:

  OC  — Ownership Compliance        (complaint / escalation buckets)
  CCS — Competitor Compliance Score  (competitor bucket)
  PSS — Product Structure Score      (product_question bucket)
  PCS — Purchase Compliance Score     (purchase_intent bucket)

Each metric is designed to show a gap exactly where the relevant
module contributes. Applied per-bucket so nothing scores 5.0 universally.

Also provides:
  severity_weighted_guard_score() — critical=0.4, high=0.25, medium=0.1
  violation_breakdown_report()    — Figure 3 in paper
  compute_third_metric()          — bucket-aware dispatcher
  compute_all_metrics()           — main entry point for ablation_runner
"""

import re
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_DEDUCTIONS = {
    "critical": 0.40,
    "high":     0.25,
    "medium":   0.10,
    "low":      0.05,
}

# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 — OC: Ownership Compliance
# Bucket: complaint, escalation
# Tests: Brand Guard + Context module (ownership language only appears when
#        guard enforces it AND context detects complaint intent)
# ─────────────────────────────────────────────────────────────────────────────

_OWNERSHIP_PATTERNS = [
    "that's on us", "thats on us", "that is on us", "this is on us",
    "our fault", "that's our fault", "we own this",
    "we failed", "we let you down",
]

_OWNERSHIP_REQUIRED_INTENTS = {
    "complaint", "product_issue", "return_refund",
    "replacement", "order_inquiry",
}

_OWNERSHIP_REQUIRED_BUCKETS = {"complaint", "escalation"}


def ownership_compliance(response: str, intent: str, bucket: str) -> Dict:
    """
    OC — binary: does response contain ownership language?
    Required only for complaint/escalation. Auto-passes for all other buckets.

    Expected gap: A=0.0, B=0.0, C=0.0, D=0.0, E≥4.0, F=5.0
    Proves: Context + Guard together drive ownership (not tone).
    """
    required = (
        intent in _OWNERSHIP_REQUIRED_INTENTS
        or bucket in _OWNERSHIP_REQUIRED_BUCKETS
    )
    if not required:
        return {"score": 1.0, "scaled": 5.0, "required": False,
                "found": False, "pattern": "n/a"}

    resp_lower = response.lower()
    for pattern in _OWNERSHIP_PATTERNS:
        if pattern in resp_lower:
            return {"score": 1.0, "scaled": 5.0, "required": True,
                    "found": True, "pattern": pattern}

    return {"score": 0.0, "scaled": 0.0, "required": True,
            "found": False, "pattern": ""}


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2 — CCS: Competitor Compliance Score
# Bucket: competitor
# Tests: Brand Guard (blocks competitor names + feature comparison)
#        + Tone (adds You vs You philosophy)
# ─────────────────────────────────────────────────────────────────────────────

_COMPETITOR_NAMES = [
    "nike", "adidas", "brooks", "hoka", "asics", "new balance",
    "saucony", "reebok", "puma", "on running", "salomon", "mizuno",
    "skechers", "under armour", "altra",
]

_PHILOSOPHY_PATTERNS = [
    "you vs you",
    "you vs. you",
    "you versus you",
    "only opponent is you",
    "only opponent is yourself",
    "beat yesterday",
    "only competition is you",
    "only competition is yourself",
    "your only competitor is you",
    "your only competitor is yourself",
    "only race is with yourself",
    "compete only with yourself",
    "competing only with yourself",
]

_FEATURE_COMPARE_WORDS = [
    "better cushioning", "more support", "superior", "outperform",
    "more cushioning", "better technology", "compared to", "better than",
    "higher quality", "more advanced", "beats", "surpasses",
]


def competitor_compliance_score(response: str, bucket: str) -> Dict:
    """
    CCS — three-component rule-based score for competitor bucket.

    Component scores (each 0 or 1):
      philosophy_present : +1 if 'You vs You' or equivalent found
      no_competitor_name : +1 if no competitor name mentioned
      no_feature_compare : +1 if no feature comparison words

    Final: mean of three components × 5 (scaled to 0-5)

    Expected gap:
      A: philosophy=0, may name competitors → CCS≈0.8-1.7
      B: no competitor names (context helps) but no philosophy → CCS≈1.7
      E: guard blocks names + features, still no philosophy → CCS≈3.3
      F: all three components → CCS=5.0
    Proves: Tone adds philosophy (E→F gap). Guard blocks violations (A→E gap).
    """
    if bucket != "competitor":
        return {"score": 1.0, "scaled": 5.0, "required": False,
                "components": {}, "detail": "n/a"}

    resp_lower = response.lower()

    # Philosophy must appear in the FIRST 10 WORDS of the response.
    # This is robust against mid-word dots (e.g. "vs.") and matches the
    # BC rubric requirement that philosophy opens the response.
    #
    # "You vs. You. That is the race." → first 10 words contain "you vs. you" ✓
    # "At Apex Stride we focus on values. You vs. You." → beyond word 10 ✗
    philosophy = False
    first_10_words = " ".join(resp_lower.split()[:10])

    for p in _PHILOSOPHY_PATTERNS:
        if p not in first_10_words:
            continue
        # Confirm phrase ends cleanly (not mid-clause)
        idx = resp_lower.find(p)
        after = resp_lower[idx + len(p):idx + len(p) + 8].strip()
        if after == "" or (after and after[0] in ".!?\n ,—:;"):
            philosophy = True
            break
    no_comp     = not any(c in resp_lower for c in _COMPETITOR_NAMES)
    no_feature  = not any(f in resp_lower for f in _FEATURE_COMPARE_WORDS)

    score  = (int(philosophy) + int(no_comp) + int(no_feature)) / 3.0
    scaled = round(score * 5, 2)

    return {
        "score":  round(score, 3),
        "scaled": scaled,
        "required": True,
        "components": {
            "philosophy_present": philosophy,
            "no_competitor_name": no_comp,
            "no_feature_compare": no_feature,
        },
        "hits": {
            "competitor_names":  [c for c in _COMPETITOR_NAMES if c in resp_lower],
            "feature_words":     [f for f in _FEATURE_COMPARE_WORDS if f in resp_lower],
            "philosophy_found":  [p for p in _PHILOSOPHY_PATTERNS if p in resp_lower],
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3 — PSS: Product Structure Score
# Bucket: product_question
# Tests: Context module (correct intent → correct response structure)
#        + Brand Guard (enforces "product first, one reason, one imperative")
#
# Why PRR failed: gpt-4o-mini already knows running products well enough
# to get them right without a context engine. The real contribution of
# context + guard is STRUCTURE — product named first, no listing, decisive.
#
# Three components, each 0 or 1:
#   product_named_first : a valid product appears in the first 10 words
#   single_product      : only one product mentioned (no listing)
#   imperative_close    : response ends with an action verb or imperative
#
# Expected gap:
#   A: no structure instruction → lists products, buries name → PSS≈1.7
#   B: context detects intent → some structure → PSS≈3.3
#   F: guard enforces "product name first. one reason. one imperative." → PSS=5.0
# ─────────────────────────────────────────────────────────────────────────────

_VALID_PRODUCTS = {
    "runner 5", "velocity x2", "ultra trail",
    "recovery stride", "marathon pro",
}

# Imperative/action verbs that close a brand-voice product response
_IMPERATIVE_CLOSERS = [
    "go.", "train.", "run.", "lace up", "move.", "earn.",
    "push.", "try it.", "get it.", "order.", "yours.", "move out",
    "make it count", "your move", "get started", "start now",
    "get yours", "go for it", "do it.", "own it.",
    "start today", "get moving", "begin today", "get out there",
]

# Hedging / listing phrases — if present, structure has failed
_LISTING_PHRASES = [
    "also consider", "you might also", "another option",
    "alternatively", "other options", "you could also try",
    "both the", "either the", "or the", "as well as the",
]


def product_structure_score(
    response: str,
    bucket: str,
) -> Dict:
    """
    PSS — measures response structure quality for product_question bucket.

    Component 1 — product_named_first (0 or 1):
      A valid product name appears within the first 10 words.
      Brand voice rule: 'Product name first.'

    Component 2 — single_product (0 or 1):
      Only one product mentioned. No listing, no alternatives.
      Brand voice rule: 'One product. One reason.'

    Component 3 — imperative_close (0 or 1):
      Response ends with an action verb or decisive statement.
      Brand voice rule: 'One imperative to close.'

    Final: mean of three components × 5 (scaled 0–5)
    """
    if bucket != "product_question":
        return {"score": 1.0, "scaled": 5.0, "required": False,
                "components": {}, "word_count": len(response.split())}

    resp_lower  = response.lower()
    words       = resp_lower.split()
    first_5     = " ".join(words[:5])

    # Component 1: product named in first 5 words
    # Brand voice: product name IS the opening word or two
    # "Runner 5. Best for beginners." scores True
    # "For beginners, the Runner 5..." scores False — product buried
    product_first = any(p in first_5 for p in _VALID_PRODUCTS)

    # Component 2: only one product mentioned
    products_found = [p for p in _VALID_PRODUCTS if p in resp_lower]
    has_listing    = any(phrase in resp_lower for phrase in _LISTING_PHRASES)
    single_product = len(products_found) == 1 and not has_listing

    # Component 3: ends with imperative (check last 15 words)
    last_15      = " ".join(words[-15:])
    imperative   = any(imp in last_15 for imp in _IMPERATIVE_CLOSERS)

    score  = (int(product_first) + int(single_product) + int(imperative)) / 3.0
    scaled = round(score * 5, 2)

    return {
        "score":    round(score, 3),
        "scaled":   scaled,
        "required": True,
        "components": {
            "product_named_first": product_first,
            "single_product":      single_product,
            "imperative_close":    imperative,
        },
        "products_found": products_found,
        "listing_hits":   [p for p in _LISTING_PHRASES if p in resp_lower],
        "word_count":     len(words),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric 4 — PCS: Purchase Compliance Score
# Bucket: purchase_intent
# Tests: Brand Guard (no corporate close, fact-only, product named, imperative)
#
# Four positive requirements (each 0 or 1):
#   product_named   : exact product name appears in response
#   fact_only       : no benefit-selling / value judgement language
#   no_corporate_close: no "is there anything else" closing questions
#   imperative_close: ends with an action verb or imperative
#
# Expected gap:
#   A: product named, but benefit-selling + no imperative → PCS≈2.5
#   E: product named + fact-only + no corporate close → PCS≈3.75
#   F: all four components → PCS=5.0
# ─────────────────────────────────────────────────────────────────────────────

# Critical forbidden phrase in brand_guard config — vanilla LLM almost
# always ends purchase responses with one of these.
_CORPORATE_CLOSE_PHRASES = [
    "is there anything else",
    "anything else i can help",
    "anything else we can help",
    "anything else i can assist",
    "anything else we can assist",
    "do you have any other questions",
    "feel free to reach out",
    "don't hesitate to contact",
    "please don't hesitate",
    "please let us know if",
    "let me know if you need",
    "if you have any questions",
    "if you need anything else",
    "happy to help with anything",
]

# Required product names for purchase intent
_PURCHASE_REQUIRED_PRODUCTS = {
    "runner 5", "velocity x2", "ultra trail",
    "recovery stride", "marathon pro",
}

# Benefit-selling / value judgement phrases — these violate fact-only requirement
_BENEFIT_SELLING_PHRASES = [
    "great value", "worth it", "worth every penny",
    "incredible", "amazing value", "best choice",
    "can't go wrong", "you'll love", "excellent",
    "fantastic", "outstanding", "superb", "top tier",
]

# Imperative closers specific to purchase intent
_IMPERATIVE_CLOSERS_PURCHASE = [
    "go.", "order.", "buy.", "get it.", "yours.",
    "lace up.", "move.", "start.", "do it.", "own it.",
    "checkout", "purchase", "add to cart",
]


def purchase_compliance_score(response: str, bucket: str) -> Dict:
    """
    PCS — measures brand guard compliance on purchase intent responses.

    Four components (each 0 or 1):
      product_named        : exact product name appears in response
      fact_only            : no benefit-selling / value judgement language
      no_corporate_close   : no closing question / "is there anything else"
      imperative_close     : ends with an action verb or imperative

    Final: mean of four components × 5 (scaled to 0-5)

    Expected gap:
      A: product named, but benefit-selling + no imperative → PCS≈2.5
      E: product named + fact-only + no corporate close → PCS≈3.75
      F: all four components → PCS=5.0
    Proves: Tone module adds imperative close (E→F gap).
            Brand Guard removes benefit-selling (A→E gap).
    """
    if bucket != "purchase_intent":
        return {"score": 1.0, "scaled": 5.0, "required": False,
                "components": {}, "word_count": len(response.split())}

    resp_lower = response.lower()
    
    # Component 1: Must explicitly name the product being purchased
    product_named = any(p in resp_lower for p in _PURCHASE_REQUIRED_PRODUCTS)
    
    # Component 2: No benefit-selling or value language
    fact_only = not any(p in resp_lower for p in _BENEFIT_SELLING_PHRASES)
    
    # Component 3: No corporate close
    no_close = not any(p in resp_lower for p in _CORPORATE_CLOSE_PHRASES)
    
    # Component 4: Ends with imperative (check last 10 words)
    last_10 = " ".join(resp_lower.split()[-10:])
    imperative = any(imp in last_10 for imp in _IMPERATIVE_CLOSERS_PURCHASE)
    
    # Score = average of 4 components
    score = (int(product_named) + int(fact_only) + int(no_close) + int(imperative)) / 4.0
    scaled = round(score * 5, 2)

    return {
        "score":    round(score, 3),
        "scaled":   scaled,
        "required": True,
        "components": {
            "product_named":        product_named,
            "fact_only":            fact_only,
            "no_corporate_close":   no_close,
            "imperative_close":     imperative,
        },
        "word_count": len(response.split()),
        "hits": {
            "benefit_selling": [p for p in _BENEFIT_SELLING_PHRASES if p in resp_lower],
            "corporate_close": [p for p in _CORPORATE_CLOSE_PHRASES if p in resp_lower],
            "imperative_found": [p for p in _IMPERATIVE_CLOSERS_PURCHASE if p in last_10],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bucket-aware dispatcher — picks the right third metric per bucket
# ─────────────────────────────────────────────────────────────────────────────

_BUCKET_METRIC_MAP = {
    "complaint":       "oc",
    "escalation":      "oc",
    "competitor":      "ccs",
    "product_question":"pss",
    "purchase_intent": "pcs",
    "edge_case":       None,   # BV+BC only, no third metric
}


def compute_third_metric(
    response:          str,
    bucket:            str,
    intent:            str,
    expected_contains: List[str],
) -> Dict:
    """
    Dispatch to the correct third metric for this bucket.
    Returns a normalised dict: {name, score, scaled, detail}
    """
    metric_name = _BUCKET_METRIC_MAP.get(bucket, None)

    if metric_name == "oc":
        r = ownership_compliance(response, intent, bucket)
        return {"name": "OC", "score": r["score"], "scaled": r["scaled"], "detail": r}

    elif metric_name == "ccs":
        r = competitor_compliance_score(response, bucket)
        return {"name": "CCS", "score": r["score"], "scaled": r["scaled"], "detail": r}

    elif metric_name == "pss":
        r = product_structure_score(response, bucket)
        return {"name": "PSS", "score": r["score"], "scaled": r["scaled"], "detail": r}

    elif metric_name == "pcs":
        r = purchase_compliance_score(response, bucket)
        return {"name": "PCS", "score": r["score"], "scaled": r["scaled"], "detail": r}

    else:
        # edge_case or unknown: return neutral 5.0, won't affect composite
        return {"name": "N/A", "score": 1.0, "scaled": 5.0, "detail": {}}


# ─────────────────────────────────────────────────────────────────────────────
# Severity-weighted guard score (used internally, not as third metric)
# ─────────────────────────────────────────────────────────────────────────────

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
        "score":           score,
        "total_deducted":  round(total_deducted, 3),
        "by_severity":     by_severity,
        "by_category":     by_category,
        "violation_count": len(violations),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point for ablation_runner
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    response:          str,
    intent:            str,
    bucket:            str,
    expected_contains: Optional[List[str]] = None,
    violations:        Optional[List[Dict]] = None,
) -> Dict:
    """
    Compute BV+BC are handled by LLM evaluator.
    This returns the rule-based third metric (bucket-aware) + SWG.

    Returns dict with:
      third_metric_name  : "OC" | "CCS" | "PSS" | "PCS" | "N/A"
      third_metric_score : 0-5 (scaled)
      third_metric_detail: raw metric output
      swg                : severity-weighted guard 0-5
      swg_detail         : raw SWG output
      word_count         : int (for reporting)
    """
    expected_contains = expected_contains or []
    violations        = violations or []

    third = compute_third_metric(response, bucket, intent, expected_contains)
    swg   = severity_weighted_guard_score(violations)

    return {
        "third_metric_name":   third["name"],
        "third_metric_score":  third["scaled"],
        "third_metric_detail": third["detail"],
        "swg":                 round(swg["score"] * 5, 2),
        "swg_detail":          swg,
        "word_count":          len(response.split()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Violation breakdown report — Figure 3 in paper
# ─────────────────────────────────────────────────────────────────────────────

def violation_breakdown_report(all_results: List[Dict]) -> Dict:
    total_responses = len(all_results)
    by_category:  Dict[str, int] = {}
    by_severity:  Dict[str, int] = {}
    by_condition: Dict[str, Dict] = {}
    by_bucket:    Dict[str, Dict] = {}
    clean_count = 0

    for r in all_results:
        viols = r.get("brand_validation", {}).get("violations", [])
        cond  = r.get("condition", "?")
        buck  = r.get("bucket", "?")

        if not viols:
            clean_count += 1

        if cond not in by_condition:
            by_condition[cond] = {"total_violations": 0, "responses": 0,
                                  "by_category": {}}
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
            by_condition[cond]["by_category"][cat] = \
                by_condition[cond]["by_category"].get(cat, 0) + 1
            by_bucket[buck]["total_violations"] += 1

    by_category_sorted = dict(
        sorted(by_category.items(), key=lambda x: x[1], reverse=True))

    return {
        "total_responses":  total_responses,
        "clean_responses":  clean_count,
        "clean_rate":       round(clean_count / max(total_responses, 1), 3),
        "by_category":      by_category_sorted,
        "by_severity":      by_severity,
        "by_condition":     by_condition,
        "by_bucket":        by_bucket,
        "top_3_categories": list(by_category_sorted.keys())[:3],
    }


def print_violation_report(report: Dict):
    print(f"\n{'='*60}")
    print("VIOLATION BREAKDOWN REPORT")
    print(f"{'='*60}")
    print(f"Total responses : {report['total_responses']}")
    print(f"Clean responses : {report['clean_responses']} "
          f"({report['clean_rate']:.0%})")
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
        print(f"  Condition {cond}: {d['total_violations']:3d} violations "
              f"across {d['responses']} responses ({rate:.1f}/response)")
    print(f"{'='*60}\n")