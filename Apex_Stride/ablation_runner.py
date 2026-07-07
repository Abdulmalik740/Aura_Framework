"""
ablation_runner.py 
===============================================
Metrics:
  BV  — Brand Voice Score          (LLM-judged, gpt-5.1)
  BC  — Behavioral Compliance      (LLM-judged, gpt-5.1)
  3rd — Bucket-specific rule-based metric:
        OC  = Ownership Compliance   (complaint / escalation)
        CCS = Competitor Compliance  (competitor)
        PRR = Product Rec Relevance  (product_question)
        PCS = Purchase Compliance    (purchase_intent)
  SWG — Severity-Weighted Guard     (rule-based)
"""

import os, json, time, argparse, math
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from openai import OpenAI
from dotenv import load_dotenv

from ablation_framework import AblationFramework, ABLATION_CONDITIONS, client
from test_suite import TEST_CASES, BUCKET_SUMMARY
from config_loader import cfg, build_brand_ethos, brand_guard_config
from metrics import (
    violation_breakdown_report,
    print_violation_report,
    compute_all_metrics,
)

load_dotenv()

# ── Forbidden phrases (for LLM evaluator context) ────────────────────────────
try:
    _guard_cfg = brand_guard_config()
    _FORBIDDEN = [p for cat in _guard_cfg["forbidden_patterns"].values()
                  for p in cat["phrases"]]
    _COMPETITORS = _guard_cfg["forbidden_patterns"]["competitor_mentions"]["phrases"]
except Exception:
    _FORBIDDEN = [
        "we're excited to announce", "valued customer", "discover your potential",
        "join the journey", "you got this", "push beyond", "push your limits",
        "find what feels right", "thank you for contacting",
        "we apologize for any inconvenience", "is there anything else i can help",
        "crush it", "you're amazing", "starting strong",
    ]
    _COMPETITORS = ["nike","adidas","brooks","hoka","asics","new balance",
                    "saucony","reebok","puma","on running","salomon","mizuno"]

WORD_LIMITS = {
    "complaint": 35, "product_issue": 35, "return_refund": 35, "replacement": 35,
    "motivation_seeking": 40, "training_question": 40, "product_question": 30,
    "sizing_help": 30, "purchase_intent": 30, "competition_comparison": 30,
    "praise": 20, "closing": 20, "order_inquiry": 40, "default": 40,
}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator — BV + BC (LLM-judged, gpt-5.1) + bucket-specific 3rd metric (rule-based)
# Metrics: OC (complaint/escalation), CCS (competitor), PRR (product_question), PCS (purchase_intent)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    brand_voice_score:       float   # LLM 1-5
    behavioral_compliance:   float   # LLM 1-5
    third_metric_name:       str     # OC / CCS / PRR / PCS / N/A
    third_metric_score:      float   # rule-based 0-5 (bucket-specific)
    swg:                     float   # severity-weighted guard 0-5
    total_score:             float   # weighted composite
    forbidden_hits:          List[str]
    competitor_hits:         List[str]
    word_count:              int
    reasoning:               str
    third_metric_detail:     Dict


class AblationEvaluator:
    """
    3-metric evaluator: BV + BC (LLM-judged, gpt-5.1) + bucket-specific 3rd metric (rule-based).
    Runs LLM eval 3× and averages for inter-run reliability.
    """
    EVAL_MODEL       = "gpt-5.1"  
    EVAL_TEMPERATURE = 0.1
    EVAL_REPEATS     = 3

    def __init__(self, client: OpenAI):
        self.client = client

    def evaluate(self, message: str, response: str, test_case: Dict) -> EvalResult:
        intent     = test_case.get("intent", "default")
        bucket     = test_case.get("bucket", "default")
        expected   = test_case.get("expected_contains", [])
        violations = test_case.get("_violations", [])

        # Rule-based bucket-aware third metric
        m = compute_all_metrics(response, intent, bucket, expected, violations)

        # Forbidden/competitor hits for reporting
        resp_lower = response.lower()
        forbidden_hits  = [p for p in _FORBIDDEN  if p in resp_lower]
        competitor_hits = [c for c in _COMPETITORS if c in resp_lower]

        # LLM eval averaged over EVAL_REPEATS
        bv_scores, bc_scores, reasonings = [], [], []
        for _ in range(self.EVAL_REPEATS):
            bv, bc, reason = self._llm_eval(message, response, test_case)
            bv_scores.append(bv)
            bc_scores.append(bc)
            reasonings.append(reason)

        bv = round(sum(bv_scores) / len(bv_scores), 2)
        bc = round(sum(bc_scores) / len(bc_scores), 2)
        tm = m["third_metric_score"]

        # Composite: BV 35% + BC 35% + ThirdMetric 30%
        # edge_case has no third metric — BV 40% + BC 60%
        if m["third_metric_name"] == "N/A":
            total = round(bv * 0.40 + bc * 0.60, 2)
        else:
            total = round(bv * 0.35 + bc * 0.35 + tm * 0.30, 2)

        return EvalResult(
            brand_voice_score=bv,
            behavioral_compliance=bc,
            third_metric_name=m["third_metric_name"],
            third_metric_score=tm,
            swg=m["swg"],
            total_score=total,
            forbidden_hits=forbidden_hits,
            competitor_hits=competitor_hits,
            word_count=m["word_count"],
            reasoning=" | ".join(dict.fromkeys(reasonings)),
            third_metric_detail=m["third_metric_detail"],
        )

    def _llm_eval(self, message: str, response: str,
                  test_case: Dict):
        intent   = test_case.get("intent", "unknown")
        trap     = test_case.get("trap", "")
        expected = test_case.get("expected_contains", [])
        not_exp  = test_case.get("expected_not_contains", [])
        brand    = build_brand_ethos().get("name", "Apex Stride")

        prompt = f"""You are evaluating a customer service response for {brand}.
Personality: tough coach. Direct, short, action-oriented. Fragments OK, contractions required.
Ownership language: 'That's on us.' — NOT 'We apologize for any inconvenience'.
Forbidden: corporate greetings, competitor names, platitudes, weak hedging.

USER MESSAGE: "{message}"
DETECTED INTENT: {intent}
KNOWN TRAP (weak system behaviour): {trap}
EXPECTED CONTENT: {expected if expected else 'N/A'}
MUST NOT CONTAIN: {not_exp if not_exp else 'N/A'}

RESPONSE TO EVALUATE:
"{response}"
evaluation:
dimensions:

```
brand_voice:
  name: "Brand Voice"
  scale: "1-5"

  hard_caps:
    - score = 1 if competitor name appears OR two_or_more_forbidden_phrases
    - score <= 2 if one_forbidden_phrase
    - score <= 3 if response_is_all_full_sentences (no fragments at all)
    - score <= 3 if word_count > 30
    - score <= 3 if benefit_selling_language
    - score <= 3 if no_contractions_used (you are / do not instead of you're / don't)

  CRITICAL: A response with ONLY full polished sentences and NO fragments CANNOT
  score above 3 regardless of content. Fragments like "Lace up." "Your move."
  "That's it." are REQUIRED for score 4 or 5.

  scoring:
    5: Pure coach voice. Mix of fragments AND contractions AND imperatives. Under 20 words.
    4: Mostly on-brand — has fragments OR contractions, one minor slip.
    3: No fragments, all full sentences. Directionally correct but sounds generic.
    2: Generic polished customer-service tone. No brand voice.
    1: Competitor name, 2+ forbidden phrases, or completely wrong behavior.

behavioral_compliance:
  name: "Behavioral Compliance"
  scale: "1-5"

  complaint:
    required:
      - ownership_first
      - resolution_next
    hard_caps:
      - score <= 2 if ownership_missing

  product_issue:
    required:
      - ownership_first
      - resolution_next
    hard_caps:
      - score <= 2 if ownership_missing

  competition_comparison:
    MANDATORY_PRE_CHECK (execute steps in order, stop at first match):
      STEP_1: Does response contain a competitor brand name?
              (nike / adidas / brooks / hoka / asics / new balance / saucony)
              IF YES → score = 1. Do not continue.
      STEP_2: Is the VERY FIRST sentence a philosophy statement?
              Philosophy sentences: "You vs. You." / "The only opponent is you."
              / "Beat yesterday." / "Only competition is yourself."
              Count words before checking — if ANY non-philosophy words come
              before the philosophy phrase, it is NOT sentence 1.
              IF NO PHILOSOPHY AT ALL → score = 2.
              IF PHILOSOPHY EXISTS BUT IS NOT SENTENCE 1 → score = 3.
      STEP_3: Philosophy IS sentence 1. No competitor names. No feature words.
              Check for feature comparison (better/more/superior/outperform).
              IF feature comparison present → score = 3.
              ELSE → eligible for 4 or 5.
      STEP_4: Score 5 if: philosophy sentence 1 + imperative verb + under 20 words.
              Score 4 if: philosophy sentence 1 + minor issue (slightly long or
              missing imperative but otherwise clean).

  product_question:
    required:
      - correct_product_first
      - one_reason
      - one_imperative

  motivation_seeking:
    required:
      - one_concrete_physical_action
      - no_platitudes

  purchase_intent:
    required:
      - state_fact_only
      - no_pushy_language

  escalation:
    required:
      - own_failure_one_sentence
      - state_fix
      - ignore_legal_threats

  scoring:
    5: All required elements present and correctly ordered.
    4: All required elements present with minor weakness.
    3: Missing one required element.
    2: Major deviation.
    1: Wrong behavior, hallucinated product, or policy failure.
```

Return ONLY valid JSON:
{{"brand_voice_score":<1-5>,"behavioral_compliance":<1-5>,"reasoning":"<20 words>"}}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.EVAL_MODEL,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON. Use full 1-5 range."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=self.EVAL_TEMPERATURE,
                max_completion_tokens=150,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            bv = max(1.0, min(5.0, float(data.get("brand_voice_score", 3))))
            bc = max(1.0, min(5.0, float(data.get("behavioral_compliance", 3))))
            return float(bv), float(bc), data.get("reasoning", "")
        except Exception as e:
            print(f"    [eval error] {e}")
            return 3.0, 3.0, "eval_failed"


# ─────────────────────────────────────────────────────────────────────────────
# Baseline runner
# ─────────────────────────────────────────────────────────────────────────────

class KeywordRAGRetriever:
    """
    Deterministic keyword-based retrieval from brand_knowledge.md.
    Sections delimited by [SECTION_NAME] headers.
    Returns max 2 matched sections per query.
    """

    def __init__(self, knowledge_base_path: str):
        self._sections: Dict[str, str] = {}
        self._load(knowledge_base_path)

    def _load(self, path: str):
        try:
            with open(path, encoding="utf-8") as f:
                raw = f.read()
        except FileNotFoundError:
            return
        current_section = None
        current_lines: List[str] = []
        for line in raw.splitlines():
            if line.startswith("[") and line.endswith("]"):
                if current_section:
                    self._sections[current_section] = "\n".join(current_lines).strip()
                current_section = line[1:-1]
                current_lines = []
            elif current_section:
                current_lines.append(line)
        if current_section:
            self._sections[current_section] = "\n".join(current_lines).strip()

    def retrieve(self, message: str, triggers: Dict[str, List[str]],
                 max_sections: int = 2) -> str:
        msg_lower = message.lower()
        matched: List[str] = []
        for section_name, keywords in triggers.items():
            if any(kw in msg_lower for kw in keywords):
                if section_name in self._sections:
                    matched.append(section_name)
        matched = matched[:max_sections]
        if not matched:
            matched = [list(self._sections.keys())[0]] if self._sections else []
        retrieved = []
        for section_name in matched:
            retrieved.append(f"=== {section_name} ===")
            retrieved.append(self._sections[section_name])
        return "\n\n".join(retrieved)


class BaselineRunner:
    def __init__(self, client: OpenAI):
        self.client      = client
        self._baselines  = cfg("baselines", default={})
        # Pre-load RAG retriever if knowledge base path is configured
        self._retrievers: Dict[str, KeywordRAGRetriever] = {}
        for bid, bcfg in self._baselines.items():
            kb_path = bcfg.get("knowledge_base_path")
            if kb_path:
                self._retrievers[bid] = KeywordRAGRetriever(kb_path)

    def generate_all(self, message: str) -> Dict[str, str]:
        results = {}
        for bid, bcfg in self._baselines.items():
            try:
                results[bid] = self._generate(message, bcfg, bid)
            except Exception as e:
                results[bid] = f"[Error: {e}]"
        return results

    def _generate(self, message: str, bcfg: Dict, bid: str = "") -> str:
        system_prompt = bcfg.get("system_prompt", "")
        template      = bcfg.get("system_prompt_template", "")

        if template and "{retrieved_context}" in template:
            retriever = self._retrievers.get(bid)
            triggers  = bcfg.get("retrieval_triggers", {})
            if retriever and triggers:
                retrieved = retriever.retrieve(message, triggers)
            else:
                retrieved = "(no relevant brand knowledge retrieved)"
            system_prompt = template.replace("{retrieved_context}", retrieved)

        messages = [{"role": "system", "content": system_prompt}]
        for ex in bcfg.get("examples", []):
            messages += [
                {"role": "user",      "content": ex["user"]},
                {"role": "assistant", "content": ex["assistant"]},
            ]
        messages.append({"role": "user", "content": message})
        resp = self.client.chat.completions.create(
            model=bcfg.get("model", "gpt-4o-mini"),
            messages=messages,
            temperature=bcfg.get("temperature", 0.7),
            max_tokens=bcfg.get("max_tokens", 200),
        )
        return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(
    conditions: Optional[List[str]] = None,
    case_ids:   Optional[List[str]] = None,
    dry_run:    bool = False,
    output_dir: str = ".",
) -> Dict:
    framework = AblationFramework(client)
    evaluator = AblationEvaluator(client)
    baselines = BaselineRunner(client)

    conditions = conditions or list(ABLATION_CONDITIONS.keys())
    test_cases = [t for t in TEST_CASES
                  if case_ids is None or t["id"] in case_ids]

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY — EMNLP 2026")
    print(f"Metrics: BV=Brand Voice  BC=Behavioral Compliance")
    print(f"         3rd metric (bucket-specific): OC=Ownership(complaint/escalation)")
    print(f"           CCS=Competitor Compliance  PRR=Product Rec Relevance  PCS=Purchase Compliance")
    print(f"Conditions : {conditions}")
    print(f"Test cases : {len(test_cases)}")
    print(f"{'='*70}\n")

    raw_results  = []
    baseline_raw = []
    total_calls  = len(conditions) * len(test_cases)
    call_idx     = 0

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
                resp = framework.run_condition(
                    tc["message"], sid, cond,
                )
                # Pass violations into evaluator for SWG
                tc_eval = dict(tc)
                tc_eval["_violations"] = resp.brand_validation.get("violations", [])
                ev = evaluator.evaluate(tc["message"], resp.response_text, tc_eval)

                print(f" → BV={ev.brand_voice_score:.1f} "
                      f"BC={ev.behavioral_compliance:.1f} "
                      f"{ev.third_metric_name}={ev.third_metric_score:.1f} "
                      f"Total={ev.total_score:.2f} "
                      f"({resp.latency_s:.1f}s)")

                raw_results.append({
                    "case_id":   tc["id"],
                    "bucket":    tc["bucket"],
                    "intent":    tc["intent"],
                    "message":   tc["message"],
                    "condition": cond,
                    "condition_label": label,
                    "response":  resp.response_text,
                    "flags": {
                        "context":     resp.context_used,
                        "tone":        resp.tone_used,
                        "brand_guard": resp.guard_used,
                    },
                    "scores": {
                        "bv":                  ev.brand_voice_score,
                        "bc":                  ev.behavioral_compliance,
                        "third_metric_name":   ev.third_metric_name,
                        "third_metric_score":  ev.third_metric_score,
                        "swg":                 ev.swg,
                        "total":               ev.total_score,
                    },
                    "brand_validation": resp.brand_validation,
                    "component_latencies": resp.component_latencies or {},
                    "latency_s":   resp.latency_s,
                    "tokens":      resp.tokens,
                })
            except Exception as e:
                print(f" [ERROR: {e}]")

        # Baselines disabled

    if dry_run:
        print("\n[DRY RUN COMPLETE]")
        return {}

    # ── Aggregate ─────────────────────────────────────────────────────────────
    ablation_summary = _aggregate_ablation(raw_results, conditions)
    baseline_summary = {}

    violation_report = violation_breakdown_report(raw_results)

    latency_summary = _aggregate_latencies(raw_results)

    # ── Context engine tracker stats ──────────────────────────────────────────
    ctx_stats  = framework.ctx_tracker.report()
    tone_stats = framework.tone_tracker.to_dict()

    # ── Save ──────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save(raw_results,        f"{output_dir}/ablation_results_raw_{ts}.json")
    _save(ablation_summary,   f"{output_dir}/ablation_summary_table_{ts}.json")
    _save(violation_report,   f"{output_dir}/violation_breakdown_{ts}.json")
    _save(latency_summary,    f"{output_dir}/component_latencies_{ts}.json")
    _save({"context_engine": ctx_stats, "tone_consistency": tone_stats},
                              f"{output_dir}/instrumentation_stats_{ts}.json")

    # ── Print tables ──────────────────────────────────────────────────────────
    _print_ablation_table(ablation_summary, conditions)
    # baseline table skipped
    print_violation_report(violation_report)
    _print_latency_table(latency_summary)
    _print_instrumentation(ctx_stats, tone_stats)

    return {
        "ablation_summary":  ablation_summary,
        "baseline_summary":  baseline_summary,
        "violation_report":  violation_report,
        "latency_summary":   latency_summary,
        "ctx_stats":         ctx_stats,
        "tone_stats":        tone_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats(vals):
    if not vals: return {"mean": 0.0, "std": 0.0, "n": 0}
    n    = len(vals)
    mean = sum(vals) / n
    std  = math.sqrt(sum((x-mean)**2 for x in vals)/n) if n > 1 else 0.0
    return {"mean": round(mean, 3), "std": round(std, 3), "n": n}


def _aggregate_ablation(raw: List[Dict], conditions: List[str]) -> Dict:
    from collections import defaultdict
    by_cond = defaultdict(lambda: defaultdict(list))
    for r in raw:
        c = r["condition"]
        s = r["scores"]
        for k in ("bv","bc","third_metric_score","swg","total"):
            by_cond[c][k].append(s.get(k, 0.0))

    by_cond_bucket = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in raw:
        by_cond_bucket[r["condition"]][r["bucket"]]["total"].append(
            r["scores"]["total"])

    summary = {}
    for cond in conditions:
        d = by_cond[cond]
        summary[cond] = {
            "label":       ABLATION_CONDITIONS[cond]["label"],
            "description": ABLATION_CONDITIONS[cond]["description"],
            "modules":     {k: ABLATION_CONDITIONS[cond][k]
                            for k in ("context","tone","brand_guard")},
            "overall": {
                "brand_voice":          _stats(d["bv"]),
                "behavioral_compliance":_stats(d["bc"]),
                "third_metric":         _stats(d["third_metric_score"]),
                "severity_guard":       _stats(d["swg"]),
                "composite":            _stats(d["total"]),
            },
            "by_bucket": {
                bucket: _stats(by_cond_bucket[cond][bucket]["total"])
                for bucket in BUCKET_SUMMARY
            },
        }
    return summary


def _aggregate_baselines(raw: List[Dict]) -> Dict:
    from collections import defaultdict
    by_bl = defaultdict(lambda: defaultdict(list))
    for r in raw:
        b = r["baseline"]
        s = r["scores"]
        for k in ("bv","bc","third_metric_score","swg","total"):
            by_bl[b][k].append(s.get(k, 0.0))
    return {
        bl: {"overall": {
            "brand_voice":          _stats(d["bv"]),
            "behavioral_compliance":_stats(d["bc"]),
            "third_metric":         _stats(d["third_metric_score"]),
            "composite":            _stats(d["total"]),
        }}
        for bl, d in by_bl.items()
    }


def _aggregate_latencies(raw: List[Dict]) -> Dict:
    from collections import defaultdict
    by_cond: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    for r in raw:
        cl = r.get("component_latencies", {})
        for comp, ms in cl.items():
            if ms is not None:
                by_cond[r["condition"]][comp].append(ms)
    return {
        cond: {comp: _stats(vals)
               for comp, vals in comps.items()}
        for cond, comps in by_cond.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_ablation_table(summary: Dict, conditions: List[str]):
    print(f"\n{'='*85}")
    print("ABLATION RESULTS TABLE  BV=Brand Voice  BC=Behavioral Compliance  3rd=Bucket Metric")
    print("  3rd: OC=Ownership  CCS=Competitor  PRR=ProductRec  PCS=PurchaseCompliance")
    print(f"{'='*85}")
    hdr = f"{'Cond':<4} {'Label':<22} {'Ctx':<4}{'Tone':<5}{'Guard':<6}"
    hdr += f"{'BV':>5}{'BC':>5}{'3rd':>6}{'Total':>7}"
    print(hdr)
    print("─" * 85)
    for cond in conditions:
        if cond not in summary: continue
        s  = summary[cond]
        m  = s["modules"]
        ov = s["overall"]
        print(
            f"{cond:<4} {s['label']:<22} "
            f"{'✓' if m['context'] else '✗':<4}"
            f"{'✓' if m['tone']    else '✗':<5}"
            f"{'✓' if m['brand_guard'] else '✗':<6}"
            f"{ov['brand_voice']['mean']:>5.2f}"
            f"{ov['behavioral_compliance']['mean']:>5.2f}"
            f"{ov['third_metric']['mean']:>6.2f}"
            f"{ov['composite']['mean']:>7.3f}"
        )
    print(f"{'='*85}\n")


def _print_baseline_table(summary: Dict):
    if not summary: return
    print(f"\n{'='*75}")
    print("BASELINE COMPARISON TABLE")
    print(f"{'='*75}")
    print(f"{'Baseline':<25}{'BV':>5}{'BC':>5}{'3rd':>6}{'Total':>7}")
    print("─" * 70)
    for bl, data in summary.items():
        ov = data["overall"]
        print(
            f"{bl:<25}"
            f"{ov['brand_voice']['mean']:>5.2f}"
            f"{ov['behavioral_compliance']['mean']:>5.2f}"
            f"{ov['third_metric']['mean']:>6.2f}"
            f"{ov['composite']['mean']:>7.3f}"
        )
    print(f"{'='*75}\n")


def _print_latency_table(latency_summary: Dict):
    print(f"\n{'='*55}")
    print("COMPONENT LATENCY TABLE (ms avg) — Table 4")
    print(f"{'='*55}")
    print(f"{'Cond':<5} {'context':>9} {'tone':>7} {'gen':>7} {'guard':>7}")
    print("─" * 55)
    for cond in sorted(latency_summary.keys()):
        d = latency_summary[cond]
        ctx  = d.get("context",    {}).get("mean", 0)
        tone = d.get("tone",       {}).get("mean", 0)
        gen  = d.get("generation", {}).get("mean", 0)
        guard= d.get("guard",      {}).get("mean", 0)
        print(f"{cond:<5} {ctx:>9.1f} {tone:>7.1f} {gen:>7.1f} {guard:>7.1f}")
    print(f"{'='*55}\n")


def _print_instrumentation(ctx_stats: Dict, tone_stats: Dict):
    print(f"\n{'='*55}")
    print("INSTRUMENTATION STATS")
    print(f"{'='*55}")
    print(f"Context engine fallback rate : "
          f"{ctx_stats.get('fallback_rate', 0):.0%} "
          f"(target <20%)")
    print(f"Avg rule-based latency       : "
          f"{ctx_stats.get('avg_rule_latency_ms', 0):.1f} ms")
    print(f"Avg AI fallback latency      : "
          f"{ctx_stats.get('avg_ai_latency_ms', 0):.1f} ms")
    print(f"Avg intent confidence        : "
          f"{ctx_stats.get('avg_confidence', 0):.2f}")
    print(f"Tone avg shift/turn          : "
          f"{tone_stats.get('avg_shift_per_turn', 0):.3f}")
    print(f"Tone smoothing rate          : "
          f"{tone_stats.get('smoothing_rate', 0):.0%}")
    dist = tone_stats.get("profile_distribution", {})
    if dist:
        dist_str = "  ".join(f"{k}:{v:.0%}" for k, v in dist.items())
        print(f"Tone profile distribution    : {dist_str}")
    print(f"{'='*55}\n")


def _save(data, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMNLP 2026 ablation study")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--cases",      nargs="+", default=None)
    parser.add_argument("--bucket",     default=None)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    case_ids = args.cases
    if args.bucket:
        case_ids = [t["id"] for t in TEST_CASES if t["bucket"] == args.bucket]
        print(f"Bucket '{args.bucket}': {len(case_ids)} cases")

    run_ablation(
        conditions=args.conditions,
        case_ids=case_ids,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )
