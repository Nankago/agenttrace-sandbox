from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenttrace_sandbox.config import AgentConfig
from agenttrace_sandbox.llm import ChatModel
from agenttrace_sandbox.runner import run_task
from agenttrace_sandbox.tracing import read_jsonl


@dataclass(frozen=True)
class ManifestRunSummary:
    total: int
    ran: int
    skipped: int
    output_path: Path
    outcomes: dict[str, int]


def run_manifest(
    manifest_path: Path,
    config: AgentConfig,
    model: ChatModel,
    output_path: Path,
    limit: int | None = None,
    dry_run: bool = False,
) -> ManifestRunSummary:
    rows = list(read_jsonl(manifest_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ran = 0
    skipped = 0
    outcomes: dict[str, int] = {}

    with output_path.open("w", encoding="utf-8") as out:
        for index, row in enumerate(rows):
            if limit is not None and ran >= limit:
                skipped += max(0, len(rows) - index)
                break
            repo = row.get("repo")
            task = row.get("task")
            if not repo or not task:
                skipped += 1
                write_record(out, row, {"ok": False, "skipped": True, "reason": "missing repo or task"})
                continue

            repo_path = Path(str(repo)).expanduser()
            if not repo_path.is_absolute():
                repo_path = (manifest_path.parent / repo_path).resolve()
            test_command = str(row.get("test_command") or "python3 -m unittest discover -s tests")

            if dry_run:
                skipped += 1
                write_record(out, row, {"ok": repo_path.exists(), "skipped": True, "repo_resolved": str(repo_path)})
                continue
            if not repo_path.exists():
                skipped += 1
                write_record(out, row, {"ok": False, "skipped": True, "reason": "repo does not exist", "repo_resolved": str(repo_path)})
                continue

            try:
                result = run_task(repo_path, str(task), test_command, config, model)
                ran += 1
                outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1
                write_record(
                    out,
                    row,
                    {
                        "ok": result.outcome == "success",
                        "skipped": False,
                        "outcome": result.outcome,
                        "failure_attribution": failure_attribution(row, result.outcome),
                        "run_id": result.run_id,
                        "trace_path": str(result.trace_path),
                        "workspace": str(result.workspace),
                        "steps": result.steps,
                        "summary": result.final_summary,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                ran += 1
                outcomes["runner_error"] = outcomes.get("runner_error", 0) + 1
                write_record(out, row, {"ok": False, "skipped": False, "outcome": "runner_error", "error": f"{type(exc).__name__}: {exc}"})

    return ManifestRunSummary(total=len(rows), ran=ran, skipped=skipped, output_path=output_path, outcomes=outcomes)


def write_record(handle, source: dict[str, Any], result: dict[str, Any]) -> None:
    handle.write(json.dumps({"source": source, "result": result}, ensure_ascii=False) + "\n")


def failure_attribution(row: dict[str, Any], outcome: str) -> str:
    if outcome == "success":
        return "passed"
    if row.get("source") != "unit_completion":
        return "unknown"
    if row.get("baseline_checked") and not row.get("baseline_ok"):
        return "baseline_failure"
    if outcome == "test_failed":
        return "model_or_task_failure"
    return "unknown"
