"""
Performance Tracker
All pricing models loaded from brand_config.yaml.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from datetime import datetime
from config_loader import llm as get_llm_cfg


@dataclass
class StepMetrics:
    step_name: str
    start_time: float
    end_time: float
    duration: float
    model: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency: float = 0.0

    def to_dict(self):
        result = {"step_name": self.step_name, "duration_seconds": round(self.duration, 3)}
        if self.model:
            result.update({
                "model": self.model,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "cost_usd": round(self.cost, 6),
                "latency_seconds": round(self.latency, 3),
            })
        else:
            result.update({"model": None, "total_tokens": 0, "cost_usd": 0.0})
        return result


@dataclass
class PipelineMetrics:
    test_case_name: str
    start_time: float
    end_time: Optional[float] = None
    steps: List[StepMetrics] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return (self.end_time - self.start_time) if self.end_time else 0

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.steps)

    @property
    def total_cost(self) -> float:
        return sum(s.cost for s in self.steps)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def total_completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.steps)

    def to_dict(self):
        return {
            "test_case_name": self.test_case_name,
            "total_duration_seconds": round(self.total_duration, 3),
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "steps": [s.to_dict() for s in self.steps],
            "step_count": len(self.steps),
        }


class PerformanceTracker:
    """Tracks performance metrics. Pricing loaded from brand_config.yaml."""

    def __init__(self):
        self._llm = get_llm_cfg()
        self._pricing = self._llm["pricing"]
        self._default_model = self._llm["default_pricing_model"]
        self.current_pipeline: Optional[PipelineMetrics] = None
        self.all_pipelines: List[PipelineMetrics] = []

    def start_pipeline(self, test_case_name: str):
        self.current_pipeline = PipelineMetrics(test_case_name=test_case_name, start_time=time.time())

    def end_pipeline(self):
        if self.current_pipeline:
            self.current_pipeline.end_time = time.time()
            self.all_pipelines.append(self.current_pipeline)
            result = self.current_pipeline
            self.current_pipeline = None
            return result
        return None

    def track_api_call(self, step_name: str, model: str, response) -> StepMetrics:
        end_time = time.time()
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)
        step = StepMetrics(step_name=step_name, start_time=0, end_time=end_time, duration=0,
                           model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                           total_tokens=total_tokens, cost=cost, latency=0.1)
        if self.current_pipeline:
            self.current_pipeline.steps.append(step)
        return step

    def track_non_api_step(self, step_name: str) -> StepMetrics:
        step = StepMetrics(step_name=step_name, start_time=0, end_time=0, duration=0)
        if self.current_pipeline:
            self.current_pipeline.steps.append(step)
        return step

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = self._pricing.get(model, self._pricing.get(self._default_model, {"prompt": 0, "completion": 0}))
        return (prompt_tokens / 1_000_000) * pricing["prompt"] + (completion_tokens / 1_000_000) * pricing["completion"]

    def get_summary(self) -> Dict:
        if not self.all_pipelines:
            return {"total_test_cases": 0, "total_duration_seconds": 0, "total_tokens": 0,
                    "total_cost_usd": 0, "average_duration_per_case": 0,
                    "average_tokens_per_case": 0, "average_cost_per_case": 0, "pipelines": []}

        n = len(self.all_pipelines)
        total_duration = sum(p.total_duration for p in self.all_pipelines)
        total_tokens = sum(p.total_tokens for p in self.all_pipelines)
        total_cost = sum(p.total_cost for p in self.all_pipelines)
        return {
            "total_test_cases": n,
            "total_duration_seconds": round(total_duration, 3),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "average_duration_per_case": round(total_duration / n, 3),
            "average_tokens_per_case": round(total_tokens / n, 2),
            "average_cost_per_case": round(total_cost / n, 6),
            "pipelines": [p.to_dict() for p in self.all_pipelines],
        }

    def save_report(self, filename: str = "performance_report.json"):
        report = {"generated_at": datetime.now().isoformat(), "summary": self.get_summary()}
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        return filename

    def print_summary(self):
        summary = self.get_summary()
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 80)
        if summary["total_test_cases"] == 0:
            print("⚠️  No pipelines tracked yet.")
            print("=" * 80 + "\n")
            return
        print(f"\n🎯 Overall Metrics:")
        print(f"   Total Test Cases:        {summary['total_test_cases']}")
        print(f"   Total Duration:          {summary['total_duration_seconds']:.2f}s")
        print(f"   Total Tokens:            {summary['total_tokens']:,}")
        print(f"   Total Cost:              ${summary['total_cost_usd']:.6f}")
        print(f"\n📈 Averages per Test Case:")
        print(f"   Duration:                {summary['average_duration_per_case']:.2f}s")
        print(f"   Tokens:                  {summary['average_tokens_per_case']:,.0f}")
        print(f"   Cost:                    ${summary['average_cost_per_case']:.6f}")
        print("\n" + "=" * 80 + "\n")


class StepTimer:
    def __init__(self, tracker: PerformanceTracker, step_name: str, is_api_call: bool = False):
        self.tracker = tracker
        self.step_name = step_name
        self.is_api_call = is_api_call
        self.start_time = None
        self.step_metrics = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"⏱️  Starting: {self.step_name}")
        if not self.is_api_call:
            self.step_metrics = self.tracker.track_non_api_step(self.step_name)
        return self

    def set_metrics(self, metrics: StepMetrics):
        self.step_metrics = metrics

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        print(f"✅ Completed: {self.step_name} ({duration:.2f}s)")
        if self.step_metrics:
            self.step_metrics.start_time = self.start_time
            self.step_metrics.duration = duration
        elif self.tracker.current_pipeline and self.tracker.current_pipeline.steps:
            for step in reversed(self.tracker.current_pipeline.steps):
                if step.step_name == self.step_name and step.duration == 0:
                    step.start_time = self.start_time
                    step.duration = duration
                    break
