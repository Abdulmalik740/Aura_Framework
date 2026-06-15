"""
instrumentation.py — EMNLP 2026 Industry Track
===============================================
Three lightweight trackers that bolt onto existing modules
without modifying their core logic:

  ContextEngineTracker  — AI fallback rate, per-path latency
  ToneConsistencyTracker — tone shift between turns, smoothing verification
  ComponentLatencyTracker — wall-clock per component (paper Table 4)

Usage — wrap your existing modules:

    tracker = ComponentLatencyTracker()

    # In ablation_framework._process():
    with tracker.time("context"):
        context = self.context_engine.analyze(...)

    with tracker.time("tone"):
        tone = self.tone_adapter.select_tone(...)

    with tracker.time("guard"):
        val = self.brand_guard.validate(...)

    with tracker.time("generation"):
        response_text, tokens = self._generate(...)
"""

import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager


# ─────────────────────────────────────────────────────────────────────────────
# 1. Context Engine Tracker — AI fallback rate + per-path latency
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContextEngineStats:
    total_requests:    int   = 0
    rule_based_count:  int   = 0
    ai_fallback_count: int   = 0
    rule_latencies_ms: List[float] = field(default_factory=list)
    ai_latencies_ms:   List[float] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)

    @property
    def fallback_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.ai_fallback_count / self.total_requests

    @property
    def avg_rule_latency_ms(self) -> float:
        return _mean(self.rule_latencies_ms)

    @property
    def avg_ai_latency_ms(self) -> float:
        return _mean(self.ai_latencies_ms)

    @property
    def avg_confidence(self) -> float:
        return _mean(self.confidence_scores)

    def to_dict(self) -> Dict:
        return {
            "total_requests":    self.total_requests,
            "rule_based_count":  self.rule_based_count,
            "ai_fallback_count": self.ai_fallback_count,
            "fallback_rate":     round(self.fallback_rate, 3),
            "avg_rule_latency_ms": round(self.avg_rule_latency_ms, 1),
            "avg_ai_latency_ms":   round(self.avg_ai_latency_ms, 1),
            "avg_confidence":      round(self.avg_confidence, 3),
            "confidence_p25":      round(_percentile(self.confidence_scores, 25), 3),
            "confidence_p50":      round(_percentile(self.confidence_scores, 50), 3),
            "confidence_p75":      round(_percentile(self.confidence_scores, 75), 3),
        }

    def summary(self) -> str:
        lines = [
            "Context Engine Stats",
            f"  Total requests   : {self.total_requests}",
            f"  Rule-based       : {self.rule_based_count} "
            f"({1-self.fallback_rate:.0%})",
            f"  AI fallback      : {self.ai_fallback_count} "
            f"({self.fallback_rate:.0%}) — target <20%",
            f"  Avg rule latency : {self.avg_rule_latency_ms:.1f} ms",
            f"  Avg AI latency   : {self.avg_ai_latency_ms:.1f} ms",
            f"  Avg confidence   : {self.avg_confidence:.2f}",
        ]
        if self.fallback_rate > 0.20:
            lines.append("  ⚠️  Fallback rate >20% — review intent keyword coverage")
        else:
            lines.append("  ✓  Fallback rate within target")
        return "\n".join(lines)


class ContextEngineTracker:
    """
    Wraps ContextUnderstandingEngine.analyze() to track fallback rate
    and per-path latency without modifying the engine itself.

    Usage:
        tracker = ContextEngineTracker()
        # monkey-patch the engine:
        original = engine.analyze
        def instrumented(message, history=None, user_profile=None):
            return tracker.track(original, message, history, user_profile)
        engine.analyze = instrumented
    """

    def __init__(self):
        self.stats = ContextEngineStats()

    def track(self, original_analyze, message: str,
              history=None, user_profile=None):
        t0 = time.perf_counter()
        result = original_analyze(message, history, user_profile)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.stats.total_requests += 1
        used_ai = getattr(result, "used_ai", False)

        if used_ai:
            self.stats.ai_fallback_count += 1
            self.stats.ai_latencies_ms.append(elapsed_ms)
        else:
            self.stats.rule_based_count += 1
            self.stats.rule_latencies_ms.append(elapsed_ms)

        conf = getattr(result, "confidence", 0.7)
        self.stats.confidence_scores.append(conf)

        return result

    def reset(self):
        self.stats = ContextEngineStats()

    def report(self) -> Dict:
        return self.stats.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tone Consistency Tracker — shift between turns, smoothing verification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToneShiftRecord:
    turn_idx:      int
    session_id:    str
    prev_profile:  str
    curr_profile:  str
    ci_delta:      float   # challenge_intensity delta
    empathy_delta: float
    total_shift:   float   # L1 norm across all numeric params
    was_smoothed:  bool    # did boundary smoothing kick in?


class ToneConsistencyTracker:
    """
    Records tone parameters each turn and computes shift metrics.
    Attach to AblationFramework._process() after tone selection.

    Provides:
      avg_shift_per_turn   — overall tone stability
      max_shift            — worst-case jump
      smoothing_rate       — % of turns where smoothing fired
      profile_distribution — which profiles were selected how often
    """

    def __init__(self, max_shift_threshold: float = 0.4):
        self.max_shift_threshold = max_shift_threshold
        self.records: List[ToneShiftRecord] = []
        self._last_by_session: Dict[str, Dict] = {}
        self.profile_counts: Dict[str, int] = {}

    def record(self, session_id: str, turn_idx: int,
               tone_params, prev_tone_params=None):
        """
        Call after each tone selection.
        tone_params: ToneParameters dataclass or dict.
        """
        curr = _tone_to_dict(tone_params)
        profile = curr.get("vocabulary_style", "unknown")
        self.profile_counts[profile] = self.profile_counts.get(profile, 0) + 1

        prev = self._last_by_session.get(session_id)
        if prev is not None:
            ci_delta     = abs(curr.get("challenge_intensity", 0) -
                               prev.get("challenge_intensity", 0))
            emp_delta    = abs(curr.get("empathy", 0) - prev.get("empathy", 0))
            total_shift  = ci_delta + emp_delta  # L1 over tracked dims
            was_smoothed = total_shift > self.max_shift_threshold

            self.records.append(ToneShiftRecord(
                turn_idx=turn_idx,
                session_id=session_id,
                prev_profile=prev.get("vocabulary_style", "?"),
                curr_profile=profile,
                ci_delta=round(ci_delta, 3),
                empathy_delta=round(emp_delta, 3),
                total_shift=round(total_shift, 3),
                was_smoothed=was_smoothed,
            ))

        self._last_by_session[session_id] = curr

    @property
    def avg_shift(self) -> float:
        return _mean([r.total_shift for r in self.records])

    @property
    def max_shift(self) -> float:
        if not self.records:
            return 0.0
        return max(r.total_shift for r in self.records)

    @property
    def smoothing_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.was_smoothed) / len(self.records)

    def profile_distribution(self) -> Dict[str, float]:
        total = sum(self.profile_counts.values())
        if total == 0:
            return {}
        return {k: round(v / total, 3) for k, v in
                sorted(self.profile_counts.items(),
                       key=lambda x: x[1], reverse=True)}

    def to_dict(self) -> Dict:
        return {
            "turn_count":          len(self.records),
            "avg_shift_per_turn":  round(self.avg_shift, 3),
            "max_shift":           round(self.max_shift, 3),
            "smoothing_rate":      round(self.smoothing_rate, 3),
            "profile_distribution":self.profile_distribution(),
            "profile_counts":      self.profile_counts,
        }

    def summary(self) -> str:
        dist = self.profile_distribution()
        dist_str = "  ".join(f"{k}:{v:.0%}" for k, v in dist.items())
        lines = [
            "Tone Consistency Stats",
            f"  Turns tracked    : {len(self.records)}",
            f"  Avg shift/turn   : {self.avg_shift:.3f} "
            f"(lower=more stable)",
            f"  Max shift        : {self.max_shift:.3f} "
            f"(threshold={self.max_shift_threshold})",
            f"  Smoothing rate   : {self.smoothing_rate:.0%} "
            f"of turns hit boundary",
            f"  Profile dist     : {dist_str}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Component Latency Tracker — wall-clock per component (Table 4)
# ─────────────────────────────────────────────────────────────────────────────

class ComponentLatencyTracker:
    """
    Context manager for timing individual pipeline components.

    Usage in AblationFramework._process():

        tracker = ComponentLatencyTracker()   # one per framework instance

        with tracker.time("context"):
            context = self.context_engine.analyze(...)

        with tracker.time("tone"):
            tone = self.tone_adapter.select_tone(...)

        with tracker.time("generation"):
            response_text, tokens = self._generate(...)

        with tracker.time("guard"):
            val = self.brand_guard.validate(...)

    Results go into each AblationResponse.component_latencies dict.
    """

    COMPONENTS = ["context", "tone", "generation", "guard", "total"]

    def __init__(self):
        self._latencies: Dict[str, List[float]] = {c: [] for c in self.COMPONENTS}
        self._current_component: Optional[str] = None
        self._current_start: float = 0.0
        self._last: Dict[str, float] = {}   # most recent timings

    @contextmanager
    def time(self, component: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._latencies[component].append(elapsed_ms)
            self._last[component] = round(elapsed_ms, 2)

    def last(self) -> Dict[str, float]:
        """Most recent timing for each component (for per-request reporting)."""
        return dict(self._last)

    def averages(self) -> Dict[str, float]:
        return {
            c: round(_mean(self._latencies[c]), 2)
            for c in self.COMPONENTS
            if self._latencies[c]
        }

    def percentiles(self, p: int = 95) -> Dict[str, float]:
        return {
            c: round(_percentile(self._latencies[c], p), 2)
            for c in self.COMPONENTS
            if self._latencies[c]
        }

    def to_dict(self) -> Dict:
        avgs = self.averages()
        p95  = self.percentiles(95)
        return {
            "averages_ms": avgs,
            "p95_ms":      p95,
            "sample_count":{c: len(self._latencies[c])
                            for c in self.COMPONENTS
                            if self._latencies[c]},
        }

    def summary(self) -> str:
        avgs = self.averages()
        p95  = self.percentiles(95)
        lines = [
            "Component Latency (ms)",
            f"  {'Component':<12} {'Mean':>8} {'P95':>8}",
            "  " + "─" * 30,
        ]
        for c in self.COMPONENTS:
            if c in avgs:
                lines.append(
                    f"  {c:<12} {avgs[c]:>8.1f} {p95.get(c, 0):>8.1f}"
                )
        lines.append("")
        lines.append("  Rule-based context path: ~2–10ms expected")
        lines.append("  AI fallback path       : ~500–1500ms expected")
        lines.append("  Brand Guard (post-gen) : ~1–5ms expected")
        return "\n".join(lines)

    def reset(self):
        self._latencies = {c: [] for c in self.COMPONENTS}
        self._last = {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _percentile(vals: List[float], p: int) -> float:
    if not vals:
        return 0.0
    sorted_vals = sorted(vals)
    idx = int(math.ceil(p / 100.0 * len(sorted_vals))) - 1
    return sorted_vals[max(0, min(idx, len(sorted_vals) - 1))]


def _tone_to_dict(tone_params) -> Dict:
    if isinstance(tone_params, dict):
        return tone_params
    if hasattr(tone_params, "to_dict"):
        return tone_params.to_dict()
    if hasattr(tone_params, "__dict__"):
        return tone_params.__dict__
    return {}
