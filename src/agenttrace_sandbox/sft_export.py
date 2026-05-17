from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenttrace_sandbox.tracing import read_jsonl

SYSTEM_PROMPT = "You are a coding agent. Return exactly one safe JSON tool call for the next step."
SFT_INSTRUCTION = "Given a coding task, plan, and previous tool history, choose the next safe tool call as JSON."


def export_sft(trace_path: Path, output_path: Path, output_format: str = "jsonl") -> int:
    samples = collect_sft_samples(trace_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        write_jsonl(samples, output_path)
    elif output_format == "alpaca":
        write_alpaca(samples, output_path)
    else:
        raise ValueError(f"unsupported output format: {output_format}")
    return len(samples)


def collect_sft_samples(trace_path: Path) -> list[dict[str, Any]]:
    traces = sorted(trace_path.glob("*/trace.jsonl")) if trace_path.is_dir() else [trace_path]
    samples: list[dict[str, Any]] = []
    for trace in traces:
        task = ""
        plan = ""
        history: list[dict[str, Any]] = []
        for event in read_jsonl(trace):
            name = event.get("event")
            payload = event.get("payload", {})
            if name == "run_started":
                task = payload.get("task", "")
            elif name == "plan":
                plan = payload.get("plan", "")
            elif name == "tool_call":
                tool = payload.get("tool")
                result = payload.get("result", {})
                if tool and tool != "finish" and result.get("ok"):
                    samples.append(make_sample(task, plan, history, payload, trace))
                history.append(history_item(payload))
    return samples


def make_sample(task: str, plan: str, history: list[dict[str, Any]], payload: dict[str, Any], trace: Path) -> dict[str, Any]:
    return {
        "instruction": SFT_INSTRUCTION,
        "input": {"task": task, "plan": plan, "history": history[-6:]},
        "output": {
            "tool": payload.get("tool"),
            "arguments": payload.get("arguments", {}),
            "reason": payload.get("reason", ""),
        },
        "metadata": {"trace": str(trace), "step": payload.get("step")},
    }


def history_item(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result", {})
    return {
        "tool": payload.get("tool"),
        "arguments": payload.get("arguments", {}),
        "ok": result.get("ok"),
        "error_type": result.get("error_type", ""),
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
