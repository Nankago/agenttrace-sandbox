from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from agenttrace_sandbox.tracing import read_jsonl


@dataclass(frozen=True)
class RunStats:
    total_runs: int
    outcomes: dict[str, int]
    pass_rate: float
    avg_steps: float
    avg_elapsed_ms: float
    format_violation_rate: float
    backends: dict[str, int]

    def render(self) -> str:
        return json.dumps(
            {
                "total_runs": self.total_runs,
                "outcomes": self.outcomes,
                "pass_rate": round(self.pass_rate, 4),
                "avg_steps": round(self.avg_steps, 2),
                "avg_elapsed_ms": round(self.avg_elapsed_ms, 2),
                "format_violation_rate": round(self.format_violation_rate, 4),
                "backends": self.backends,
            },
            ensure_ascii=False,
            indent=2,
        )


@dataclass(frozen=True)
class ManifestStats:
    total: int
    ran: int
    skipped: int
    by_source: dict[str, dict[str, Any]]

    def render(self) -> str:
        return json.dumps(
            {
                "total": self.total,
                "ran": self.ran,
                "skipped": self.skipped,
                "by_source": self.by_source,
            },
            ensure_ascii=False,
            indent=2,
        )


def compute_run_stats(runs_path: Path) -> RunStats:
    traces = sorted(runs_path.glob("*/trace.jsonl")) if runs_path.is_dir() else [runs_path]
    outcomes: dict[str, int] = {}
    backends: dict[str, int] = {}
    step_counts: list[int] = []
    elapsed_values: list[float] = []
    tool_calls = 0
    format_violations = 0

    for trace in traces:
        if not trace.exists():
            continue
        run_started: dict[str, Any] = {}
        run_finished: dict[str, Any] = {}
        steps = 0
        for event in read_jsonl(trace):
            name = event.get("event")
            payload = event.get("payload", {})
            if name == "run_started":
                run_started = payload
            elif name == "tool_call":
                steps += 1
                tool_calls += 1
                if payload.get("format_violation"):
                    format_violations += 1
            elif name == "run_finished":
                run_finished = payload

        outcome = str(run_finished.get("outcome") or "missing_outcome")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        backend = str(run_started.get("sandbox_backend") or "unknown")
        backends[backend] = backends.get(backend, 0) + 1
        step_counts.append(steps)
        elapsed = run_finished.get("elapsed_ms")
        if isinstance(elapsed, (int, float)):
            elapsed_values.append(float(elapsed))

    total = sum(outcomes.values())
    success = outcomes.get("success", 0)
    return RunStats(
        total_runs=total,
        outcomes=outcomes,
        pass_rate=success / total if total else 0.0,
        avg_steps=mean(step_counts) if step_counts else 0.0,
        avg_elapsed_ms=mean(elapsed_values) if elapsed_values else 0.0,
        format_violation_rate=format_violations / tool_calls if tool_calls else 0.0,
        backends=backends,
    )


def compute_manifest_stats(results_path: Path) -> ManifestStats:
    groups: dict[str, dict[str, Any]] = {}
    total = 0
    ran = 0
    skipped = 0
    for row in read_jsonl(results_path):
        total += 1
        source = row.get("source", {})
        result = row.get("result", {})
        source_name = str(source.get("source") or "unknown")
        group = groups.setdefault(source_name, {"total": 0, "ran": 0, "skipped": 0, "success": 0, "outcomes": {}})
        group["total"] += 1
        if result.get("skipped"):
            skipped += 1
            group["skipped"] += 1
            outcome = "skipped"
        else:
            ran += 1
            group["ran"] += 1
            outcome = str(result.get("outcome") or "missing_outcome")
            if outcome == "success":
                group["success"] += 1
        outcomes = group["outcomes"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    for group in groups.values():
        group["pass_rate"] = round(group["success"] / group["ran"], 4) if group["ran"] else 0.0
    return ManifestStats(total=total, ran=ran, skipped=skipped, by_source=groups)
