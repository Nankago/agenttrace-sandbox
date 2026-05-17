from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenttrace_sandbox.tracing import read_jsonl


def export_sft(trace_path: Path, output_path: Path) -> int:
    traces = sorted(trace_path.glob("*/trace.jsonl")) if trace_path.is_dir() else [trace_path]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
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
                        out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        count += 1
                    history.append(
                        {
                            "tool": tool,
                            "arguments": payload.get("arguments", {}),
                            "ok": result.get("ok"),
                            "error_type": result.get("error_type", ""),
                        }
                    )
    return count
