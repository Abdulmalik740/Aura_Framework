"""
ablation_runner.py — Ember & Edge 
"""

import os, json, time, argparse, math
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from openai import OpenAI
from dotenv import load_dotenv

from ablation_framework import AblationFramework, ABLATION_CONDITIONS, client
from test_suite import TEST_CASES, BUCKET_SUMMARY
from config_loader import cfg, build_brand_ethos, brand_guard_config
from metrics import violation_breakdown_report, print_violation_report, compute_all_metrics

load_dotenv()

try:
    _guard_cfg = brand_guard_config()
    _FORBIDDEN = [p for cat in _guard_cfg["forbidden_patterns"].values() for p in cat["phrases"]]
except Exception:
    _FORBIDDEN = [
        "best-in-class", "industry-leading", "revolutionary", "game-changing",
        "premium quality", "buy now", "don't miss out", "limited time", "act now",
        "does that make sense", "does that help", "let me know if", "feel free to",
        "thank you for contacting", "we apologize for any inconvenience",
        "is there anything else i can help", "excellent choice",
    ]

_COMPETITORS = ["wusthof", "shun", "global", "victorinox", "henckels", "miyabi", "zwilling", "dalstrong", "misen", "tojiro"]


@dataclass
class EvalResult:
    brand_voice_score: float
    behavioral_compliance: float
    third_metric_name: str
    third_metric_score: float
    swg: float
    total_score: float
    forbidden_hits: List[str]
    competitor_hits: List[str]
    word_count: int
    reasoning: str
    third_metric_detail: Dict


class AblationEvaluator:
    EVAL_MODEL = "gpt-5.1"
    EVAL_TEMPERATURE = 0.1
    EVAL_REPEATS = 3

    def __init__(self, client: OpenAI):
        self.client = client

    def evaluate(self, message: str, response: str, test_case: Dict) -> EvalResult:
        intent = test_case.get("intent", "default")
        bucket = test_case.get("bucket", "default")
        expected = test_case.get("expected_contains", [])
        violations = test_case.get("_violations", [])

        # Rule-based third metric
        m = compute_all_metrics(response, intent, bucket, expected, violations, original_message=message)

        resp_lower = response.lower()
        
        # Expanded forbidden patterns
        FORBIDDEN_EXTENDED = _FORBIDDEN + [
            "we apologize", "we're sorry", "sorry to hear", "thank you for contacting",
            "we take this seriously", "rest assured", "please allow", "we'll look into",
            "is there anything else", "feel free to", "let me know if",
        ]
        forbidden_hits = [p for p in FORBIDDEN_EXTENDED if p in resp_lower]
        competitor_hits = [c for c in _COMPETITORS if c in resp_lower]

        # LLM eval for BV and BC
        bv_scores, bc_scores, reasonings = [], [], []
        for _ in range(self.EVAL_REPEATS):
            bv, bc, reason = self._llm_eval(message, response, test_case)
            bv_scores.append(bv)
            bc_scores.append(bc)
            reasonings.append(reason)

        bv = round(sum(bv_scores) / len(bv_scores), 2)
        bc = round(sum(bc_scores) / len(bc_scores), 2)

        # ========== RULE-BASED ADJUSTMENTS (AFTER bv/bc are assigned) ==========

        has_ownership = any(p in resp_lower for p in [
            "that's not right", "thats not right", "that is not right",
            "that shouldn't happen", "that should not happen",
            "that's on us", "this is on us", "our fault",
            "we failed you", "this isn't right",
        ])
        has_corporate = any(p in resp_lower for p in [
            "we apologize", "we're sorry", "sorry to hear",
            "thank you for contacting", "thank you for bringing",
            "we take this seriously", "rest assured", "please allow",
            "valued customer", "we appreciate your",
        ])
        has_deflection = any(p in resp_lower for p in [
            "cutting board", "cutting surface", "technique",
            "how you've been using", "care for", "the way you store",
            "check your spam", "check directly with",
        ])
        has_concrete = any(p in resp_lower for p in [
            "send", "replace", "refund", "fix", "photo", "today",
            "right away", "immediately", "now",
        ])
        has_poetry_in_complaint = any(p in resp_lower for p in [
            "culinary journey", "culinary adventure", "beautiful journey",
            "harmonious", "artistry", "craft", "companion in your",
        ])

        # 1. Complaint / escalation bucket rules
        if bucket in ["complaint", "escalation", "edge_case"]:
            # AC=0 on a required bucket → hard cap at 2.5
            # This is the main fix — prevents A/G from scoring 3+ when they fail accountability
            if not has_ownership:
                bv = min(bv, 2.5)
                bc = min(bc, 2.5)
            
            # Ownership phrase present + concrete action + no corporate = strong reward
            if has_ownership and has_concrete and not has_corporate:
                bv = min(5.0, bv + 1.2)
                bc = min(5.0, bc + 1.2)
            elif has_ownership and not has_corporate:
                bv = min(5.0, bv + 0.8)
                bc = min(5.0, bc + 0.8)
            
            # Corporate apology without ownership = penalise
            if has_corporate and not has_ownership:
                bv = max(1.0, bv - 1.0)
                bc = max(1.0, bc - 1.0)
            
            # Deflection (blaming customer/technique) = penalise
            if has_deflection and not has_ownership:
                bv = max(1.0, bv - 1.0)
            
            # Poetry in complaint context = penalise
            if has_poetry_in_complaint and not has_ownership:
                bv = max(1.0, bv - 0.5)

        # 2. Product question rules
        if bucket == "product_question":
            knife_named = any(k in resp_lower for k in [
                "artisan chef", "paring knife", "bread knife",
                "santoku", "beginner set",
            ])
            lists_multiple = sum(1 for k in [
                "artisan chef", "paring knife", "bread knife", "santoku", "beginner set",
            ] if k in resp_lower) > 1
            
            if knife_named and not lists_multiple:
                bv = min(5.0, bv + 0.5)
            if lists_multiple:
                bv = max(1.0, bv - 0.5)
                bc = max(1.0, bc - 0.5)

        # 3. Purchase intent rules
        if bucket == "purchase_intent":
            words_10 = " ".join(resp_lower.split()[:10])
            immediate = any(s in words_10 for s in ["yes", "in stock", "available"])
            
            if immediate and not has_deflection:
                bv = min(5.0, bv + 1.2)
                bc = min(5.0, bc + 0.8)
            elif has_deflection:  # "I can't check stock directly"
                bv = max(1.0, bv - 1.5)
                bc = max(1.0, bc - 1.5)

        # 4. Competitor rules
        if bucket == "competitor":
            dwells_on_comp = any(p in resp_lower for p in [
                "better than", "sharper than", "outperform",
                "compared to", "superior", "beats",
                "has a long history", "rich heritage", "storied",
            ])
            spec_compare = any(p in resp_lower for p in [
                "hrc", "rockwell", "blade angle", "steel composition",
                "ergonomic", "price point", "affordable",
            ])
            if dwells_on_comp or spec_compare:
                bv = max(1.0, bv - 1.0)
                bc = max(1.0, bc - 1.0)

        # 5. Apply guard violation penalties to BC
        if violations:
            critical_count = sum(1 for v in violations if v.get("severity") == "critical")
            high_count = sum(1 for v in violations if v.get("severity") == "high")
            if critical_count > 0:
                bc = max(1.0, bc - 1.5 * critical_count)
            if high_count > 0:
                bc = max(1.0, bc - 0.75 * high_count)

        # Use rule-based third metric
        tm = m["third_metric_score"]

        total = round(bv * 0.35 + bc * 0.35 + tm * 0.30, 2)

        return EvalResult(
            brand_voice_score=bv, behavioral_compliance=bc,
            third_metric_name=m["third_metric_name"], third_metric_score=tm,
            swg=m["swg"], total_score=total,
            forbidden_hits=forbidden_hits, competitor_hits=competitor_hits,
            word_count=m["word_count"], reasoning=" | ".join(dict.fromkeys(reasonings)),
            third_metric_detail=m["third_metric_detail"],
        )
    def _llm_eval(self, message: str, response: str, test_case: Dict):
        intent = test_case.get("intent", "unknown")
        bucket = test_case.get("bucket", "unknown")

        # Per-bucket scoring rubric aligned to Ember & Edge brand personality
        bucket_rubric = {
            "competitor": """
BUCKET: competitor comparison
BRAND MOVE: Acknowledge competitor briefly, pivot to craft/presence/ingredient philosophy, name the Artisan Chef's Knife, end with a cooking question.
5 = Brief acknowledgment + sensory/philosophy pivot + knife named + cooking question. No feature specs.
4 = Good pivot and knife named but close is weak or generic
3 = Names knife but pivot is generic ("we're different", "unique blend") with no sensory language
2 = No pivot — just describes the knife. Or uses "better than" / feature comparison language.
1 = Corporate language, tech specs, or completely ignores the comparison question""",

            "complaint": """
BUCKET: complaint / product issue
BRAND MOVE: Own it immediately ("That's not right." / "That shouldn't happen."), state the concrete fix, no poetry.
5 = Ownership phrase + concrete fix + under 60 words + no sensory/poetic language
4 = Ownership phrase present but fix is vague, or slightly too long
3 = Acknowledges problem but no ownership phrase, or fix is missing
2 = Corporate apology ("we apologize", "sorry to hear") with no ownership or fix
1 = Ignores the complaint, uses poetry, or gives generic unhelpful response""",

            "escalation": """
BUCKET: escalation (angry/threatening customer)
BRAND MOVE: Same as complaint — own it fast, fix it fast, no poetry, no defensiveness.
5 = Immediate ownership + concrete resolution + calm + under 60 words
4 = Ownership present, resolution slightly vague
3 = De-escalates but no ownership phrase or resolution
2 = Defensive or uses corporate language
1 = Ignores escalation signals or responds with poetry""",

            "product_question": """
BUCKET: product question / recommendation
BRAND MOVE: Ask what they cook (or lead with sensory question), name ONE specific knife, give one sensory reason, end with a gentle question.
5 = Names ONE knife + sensory/feel language + question about their cooking + under 65 words
4 = Names knife + question but language is generic ("versatile", "perfect for")
3 = Names knife but no question, or question but no knife named
2 = Lists multiple knives, or no knife named at all
1 = Wrong knife for context, or corporate/spec-heavy language""",

            "purchase_intent": """
BUCKET: purchase intent (customer wants to buy)
BRAND MOVE: Answer the stock question immediately, name the knife, decisive close, under 40 words. No poetry on transactional questions.
5 = "Yes, in stock" in first words + knife named + decisive close + under 40 words
4 = Answers stock question but adds slight filler, still under 50 words
3 = Answers stock question but adds poetic filler or exceeds 50 words
2 = Hedges on stock ("I can't check directly") or adds irrelevant content
1 = Doesn't answer stock question, corporate language, or pushy sales""",

            "edge_case": """
BUCKET: edge case (multi-signal: refund + legal threat + competitor mention)
BRAND MOVE: Own it, give the refund path clearly, no corporate language, no poetry.
5 = Ownership phrase + clear refund path + no corporate language + under 60 words
4 = Ownership present, refund path slightly vague
3 = Addresses refund but no ownership phrase
2 = Corporate language or defensive tone
1 = Ignores the refund request or uses generic apology""",
        }

        rubric = bucket_rubric.get(bucket, """
GENERAL SCORING:
5 = Sensory language, names a knife, decisive, natural contractions, under 70 words
4 = Good brand voice but one element missing
3 = Correct but generic — no sensory language, no specific recommendation  
2 = Corporate phrases, hedging, or spec-heavy
1 = Wrong recommendation, competitor mentioned, or completely off-brand""")

        prompt = f"""You are evaluating a customer service response for Ember & Edge — a sensory, poetic knife brand.

BRAND PERSONALITY: Patient Michelin-trained chef-instructor. Calm, poetic, sensory. Never corporate. Never pushy.
GOOD LANGUAGE: "the blade whispers through ripe tomato skin", "every cut is a moment of presence", "what do you feel when you hold it"
BAD LANGUAGE: "unique blend", "innovative design", "would you like to explore", "does that help", "thank you for contacting"

USER MESSAGE: "{message}"
RESPONSE TO EVALUATE: "{response}"

{rubric}

Score brand_voice_score (1-5): how well does the response sound like Ember & Edge — sensory, poetic where appropriate, decisive, no corporate language
Score behavioral_compliance (1-5): how well does the response follow the correct brand MOVE for this bucket

Return ONLY JSON: {{"brand_voice_score":1-5,"behavioral_compliance":1-5,"reasoning":"one sentence"}}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.EVAL_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict brand evaluator. Return ONLY valid JSON. Be accurate — a response with sensory language and correct brand move should score 4-5. A response with corporate language or wrong move should score 1-2. Do not default to 3."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.EVAL_TEMPERATURE,
                max_completion_tokens=150,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            bv = max(1, min(5, int(data.get("brand_voice_score", 3))))
            bc = max(1, min(5, int(data.get("behavioral_compliance", 3))))
            return float(bv), float(bc), data.get("reasoning", "")
        except Exception as e:
            print(f"Eval error: {e}")
            return 3.0, 3.0, "eval_error"


def run_ablation(conditions: Optional[List[str]] = None, case_ids: Optional[List[str]] = None,
                 dry_run: bool = False, output_dir: str = ".") -> Dict:
    framework = AblationFramework(client)
    evaluator = AblationEvaluator(client)

    conditions = conditions or list(ABLATION_CONDITIONS.keys())
    test_cases = [t for t in TEST_CASES if case_ids is None or t["id"] in case_ids]

    print(f"\n{'='*75}")
    print(f"ABLATION STUDY — EMBER & EDGE — EMNLP 2026")
    print(f"Conditions: {conditions} | Test cases: {len(test_cases)}")
    print(f"{'='*75}\n")

    raw_results = []
    total_calls = len(conditions) * len(test_cases)
    call_idx = 0

    for tc in test_cases:
        print(f"\n[{tc['id']}] {tc['bucket'].upper()} | {tc['intent']}")
        print(f"  {tc['message'][:80]}...")

        for cond in conditions:
            call_idx += 1
            label = ABLATION_CONDITIONS[cond]["label"]
            print(f"  [{call_idx}/{total_calls}] {cond}: {label}", end="", flush=True)

            if dry_run:
                print(" [DRY RUN]")
                continue

            sid = f"abl_{tc['id']}_{cond}_{int(time.time())}"
            try:
                # Pass bucket through to the framework for metrics
                resp = framework.run_condition(tc["message"], sid, cond, bucket=tc["bucket"])
                print(f"\n  === RESPONSE FOR {cond} ===")
                print(f"  {resp.response_text}")
                print(f"  =========================")
                tc_eval = dict(tc)
                tc_eval["_violations"] = resp.brand_validation.get("violations", [])
                ev = evaluator.evaluate(tc["message"], resp.response_text, tc_eval)

                print(f" → BV={ev.brand_voice_score:.1f} BC={ev.behavioral_compliance:.1f} {ev.third_metric_name}={ev.third_metric_score:.1f} Total={ev.total_score:.2f} ({resp.latency_s:.1f}s)")

                raw_results.append({
                    "case_id": tc["id"], "bucket": tc["bucket"], "intent": tc["intent"],
                    "message": tc["message"], "condition": cond, "condition_label": label,
                    "response": resp.response_text,
                    "flags": {"context": resp.context_used, "tone": resp.tone_used, "brand_guard": resp.guard_used},
                    "scores": {"bv": ev.brand_voice_score, "bc": ev.behavioral_compliance,
                               "third_metric_name": ev.third_metric_name, "third_metric_score": ev.third_metric_score,
                               "swg": ev.swg, "total": ev.total_score},
                    "brand_validation": resp.brand_validation,
                    "component_latencies": resp.component_latencies or {},
                    "latency_s": resp.latency_s, "tokens": resp.tokens,
                })
            except Exception as e:
                print(f" [ERROR: {e}]")
                import traceback
                traceback.print_exc()

    if dry_run:
        print("\n[DRY RUN COMPLETE]")
        return {}

    # Aggregation
    ablation_summary = _aggregate_ablation(raw_results, conditions)
    violation_report = violation_breakdown_report(raw_results)
    latency_summary = _aggregate_latencies(raw_results)
    ctx_stats = framework.ctx_tracker.report()
    tone_stats = framework.tone_tracker.to_dict()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save(raw_results, f"{output_dir}/ablation_results_raw_{ts}.json")
    _save(ablation_summary, f"{output_dir}/ablation_summary_table_{ts}.json")
    _save(violation_report, f"{output_dir}/violation_breakdown_{ts}.json")
    _save(latency_summary, f"{output_dir}/component_latencies_{ts}.json")

    _print_ablation_table(ablation_summary, conditions)
    print_violation_report(violation_report)
    _print_latency_table(latency_summary)

    return {"ablation_summary": ablation_summary, "violation_report": violation_report}


def _stats(vals):
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    n = len(vals)
    mean = sum(vals) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in vals) / n) if n > 1 else 0.0
    return {"mean": round(mean, 3), "std": round(std, 3), "n": n}


def _aggregate_ablation(raw: List[Dict], conditions: List[str]) -> Dict:
    from collections import defaultdict
    by_cond = defaultdict(lambda: defaultdict(list))
    for r in raw:
        c = r["condition"]
        s = r["scores"]
        for k in ("bv", "bc", "third_metric_score", "swg", "total"):
            by_cond[c][k].append(s.get(k, 0.0))

    summary = {}
    for cond in conditions:
        d = by_cond[cond]
        summary[cond] = {
            "label": ABLATION_CONDITIONS[cond]["label"],
            "modules": {k: ABLATION_CONDITIONS[cond][k] for k in ("context", "tone", "brand_guard")},
            "overall": {
                "brand_voice": _stats(d["bv"]),
                "behavioral_compliance": _stats(d["bc"]),
                "third_metric": _stats(d["third_metric_score"]),
                "severity_guard": _stats(d["swg"]),
                "composite": _stats(d["total"]),
            },
        }
    return summary


def _aggregate_latencies(raw: List[Dict]) -> Dict:
    from collections import defaultdict
    by_cond = defaultdict(lambda: defaultdict(list))
    for r in raw:
        cl = r.get("component_latencies", {})
        for comp, ms in cl.items():
            if ms is not None:
                by_cond[r["condition"]][comp].append(ms)
    return {cond: {comp: _stats(vals) for comp, vals in comps.items()} for cond, comps in by_cond.items()}


def _print_ablation_table(summary: Dict, conditions: List[str]):
    print(f"\n{'='*90}")
    print("ABLATION RESULTS TABLE")
    print(f"{'='*90}")
    print(f"{'Cond':<4} {'Label':<22} {'Ctx':<4}{'Tone':<5}{'Guard':<6}{'BV':>5}{'BC':>5}{'3rd':>6}{'Total':>7}")
    print("─" * 90)
    for cond in conditions:
        if cond not in summary:
            continue
        s = summary[cond]
        m = s["modules"]
        ov = s["overall"]
        print(f"{cond:<4} {s['label']:<22} {'✓' if m['context'] else '✗':<4}{'✓' if m['tone'] else '✗':<5}{'✓' if m['brand_guard'] else '✗':<6}{ov['brand_voice']['mean']:>5.2f}{ov['behavioral_compliance']['mean']:>5.2f}{ov['third_metric']['mean']:>6.2f}{ov['composite']['mean']:>7.3f}")
    print(f"{'='*90}\n")


def _print_latency_table(latency_summary: Dict):
    print(f"\n{'='*55}")
    print("COMPONENT LATENCY (ms avg)")
    print(f"{'='*55}")
    print(f"{'Cond':<5} {'context':>9} {'tone':>7} {'gen':>7} {'guard':>7}")
    print("─" * 55)
    for cond in sorted(latency_summary.keys()):
        d = latency_summary[cond]
        ctx = d.get("context", {}).get("mean", 0)
        tone = d.get("tone", {}).get("mean", 0)
        gen = d.get("generation", {}).get("mean", 0)
        guard = d.get("guard", {}).get("mean", 0)
        print(f"{cond:<5} {ctx:>9.1f} {tone:>7.1f} {gen:>7.1f} {guard:>7.1f}")
    print(f"{'='*55}\n")


def _save(data, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    case_ids = args.cases
    if args.bucket:
        case_ids = [t["id"] for t in TEST_CASES if t["bucket"] == args.bucket]
        print(f"Bucket '{args.bucket}': {len(case_ids)} cases")

    run_ablation(conditions=args.conditions, case_ids=case_ids, dry_run=args.dry_run, output_dir=args.output_dir)
