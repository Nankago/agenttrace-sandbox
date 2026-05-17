from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenttrace_sandbox.tracing import read_jsonl


def export_sft(trace_path: Path, output_path: Path, output_format: str = "jsonl") -> int:
    traces = sorted(trace_path.glob("*/trace.jsonl")) if trace_path.is_dir() else [trace_path]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    samples: list[dict[str, Any]] = []
    system = "You are a coding agent. Return exactly one safe JSON tool call for the next step."
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
                    sample = {
                        "instruction": "Given a coding task, plan, and previous tool history, choose the next safe tool call as JSON.",
                        "input": {
                            "task": task,
                            "plan": plan,
                            "history": history[-6:],
                        },
                        "output": {
                            "tool": tool,
                            "arguments": payload.get("arguments", {}),
                            "reason": payload.get("reason", ""),
                        },
                        "metadata": {
                            "trace": str(trace),
                            "step": payload.get("step"),
                        },
                    }
                    samples.append(sample)
                history.append(
                    {
                        "tool": tool,
                        "arguments": payload.get("arguments", {}),
                        "ok": result.get("ok"),
                        "error_type": result.get("error_type", ""),
                    }
                )
    if output_format == "alpaca":
        alpaca_rows = [
            {
                "system": system,
                "instruction": sample["instruction"],
                "input": json.dumps(sample["input"], ensure_ascii=False, indent=2),
                "output": json.dumps(sample["output"], ensure_ascii=False, indent=2),
            }
            for sample in samples
        ]
        output_path.write_text(json.dumps(alpaca_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(alpaca_rows)
    if output_format != "jsonl":
        raise ValueError(f"unsupported output format: {output_format}")
    with output_path.open("w", encoding="utf-8") as out:
        for sample in samples:
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count
