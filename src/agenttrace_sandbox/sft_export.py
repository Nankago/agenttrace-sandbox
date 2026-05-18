from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenttrace_sandbox.tracing import read_jsonl

SYSTEM_PROMPT = "You are a coding agent. Return exactly one safe JSON tool call for the next step."
SFT_INSTRUCTION = "Given a coding task, plan, and previous tool history, choose the next safe tool call as JSON."


def export_sft(
    trace_path: Path,
    output_path: Path,
    output_format: str = "jsonl",
    strict: bool = False,
    clean_steps: bool = False,
    reject_test_edits: bool = False,
) -> int:
    samples = collect_sft_samples(trace_path, strict=strict, clean_steps=clean_steps, reject_test_edits=reject_test_edits)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        write_jsonl(samples, output_path)
    elif output_format == "alpaca":
        write_alpaca(samples, output_path)
    else:
        raise ValueError(f"unsupported output format: {output_format}")
    return len(samples)


def collect_sft_samples(trace_path: Path, strict: bool = False, clean_steps: bool = False, reject_test_edits: bool = False) -> list[dict[str, Any]]:
    traces = sorted(trace_path.glob("*/trace.jsonl")) if trace_path.is_dir() else [trace_path]
    samples: list[dict[str, Any]] = []
    for trace in traces:
        events = list(read_jsonl(trace))
        if strict and not is_high_quality_trace(events, reject_test_edits=reject_test_edits):
            continue
        if clean_steps and not is_successful_trace(events, reject_test_edits=reject_test_edits):
            continue
        task = ""
        plan = ""
        history: list[dict[str, Any]] = []
        for event in events:
            name = event.get("event")
            payload = event.get("payload", {})
            if name == "run_started":
                task = payload.get("task", "")
            elif name == "plan":
                plan = payload.get("plan", "")
            elif name == "tool_call":
                tool = payload.get("tool")
                result = payload.get("result", {})
                if clean_steps and not is_clean_tool_call(payload):
                    history.append(history_item(payload))
                    continue
                if tool and tool != "finish" and result.get("ok"):
                    samples.append(make_sample(task, plan, history, payload, trace))
                history.append(history_item(payload))
    return samples


def is_successful_trace(events: list[dict[str, Any]], reject_test_edits: bool = False) -> bool:
    finished = next((event.get("payload", {}) for event in events if event.get("event") == "run_finished"), {})
    if finished.get("outcome") != "success":
        return False
    if reject_test_edits and diff_touches_tests(str(finished.get("diff", ""))):
        return False
    return any(
        event.get("event") == "tool_call"
        and event.get("payload", {}).get("tool") == "run_tests"
        and event.get("payload", {}).get("result", {}).get("ok")
        for event in events
    )


def is_high_quality_trace(events: list[dict[str, Any]], reject_test_edits: bool = False) -> bool:
    if not is_successful_trace(events, reject_test_edits=reject_test_edits):
        return False
    for event in events:
        if event.get("event") != "tool_call":
            continue
        payload = event.get("payload", {})
        if not is_clean_tool_call(payload):
            return False
        if reject_test_edits and tool_edits_tests(payload):
            return False
    return True


def is_clean_tool_call(payload: dict[str, Any]) -> bool:
    result = payload.get("result", {})
    return not (
        payload.get("format_violation")
        or payload.get("parse_error")
        or payload.get("retries_used")
        or result.get("blocked")
        or result.get("error_type")
        or not result.get("ok")
    )


def tool_edits_tests(payload: dict[str, Any]) -> bool:
    if payload.get("tool") not in {"replace_in_file", "write_file"}:
        return False
    path = str(payload.get("arguments", {}).get("path", ""))
    return path.startswith("tests/") or "/tests/" in path or path.startswith("test_") or path.endswith("_test.py")


def diff_touches_tests(diff: str) -> bool:
    for line in diff.splitlines():
        if not (line.startswith("--- ") or line.startswith("+++ ")):
            continue
        path = line[4:].strip()
        if path.startswith(("a/", "b/")):
            path = path[2:]
        if path.startswith("tests/") or "/tests/" in path or path.startswith("test_") or path.endswith("_test.py"):
            return True
    return False


def make_sample(task: str, plan: str, history: list[dict[str, Any]], payload: dict[str, Any], trace: Path) -> dict[str, Any]:
    return {
        "instruction": SFT_INSTRUCTION,
        "input": {"task": task, "plan": plan, "history": history[-6:]},
        "output": {
            "tool": payload.get("tool"),
            "arguments": payload.get("arguments", {}),
            "reason": payload.get("reason", ""),
        },
        "metadata": {
            "trace": str(trace),
            "step": payload.get("step"),
            "format_violation": bool(payload.get("format_violation")),
        },
    }


def history_item(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result", {})
    return {
        "tool": payload.get("tool"),
        "arguments": payload.get("arguments", {}),
        "ok": result.get("ok"),
        "error_type": result.get("error_type", ""),
        "format_violation": bool(payload.get("format_violation")),
    }


def write_jsonl(samples: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as out:
        for sample in samples:
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_alpaca(samples: list[dict[str, Any]], output_path: Path) -> None:
    rows = [
        {
            "system": SYSTEM_PROMPT,
            "instruction": sample["instruction"],
            "input": json.dumps(sample["input"], ensure_ascii=False, indent=2),
            "output": json.dumps(sample["output"], ensure_ascii=False, indent=2),
        }
        for sample in samples
    ]
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
