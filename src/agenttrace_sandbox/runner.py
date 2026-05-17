from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agenttrace_sandbox.config import AgentConfig
from agenttrace_sandbox.llm import ChatModel, extract_json_object
from agenttrace_sandbox.sandbox import Sandbox
from agenttrace_sandbox.tools import ToolRegistry, ToolResult
from agenttrace_sandbox.tracing import write_event


PLANNER_SYSTEM = "You are a careful coding-agent planner. Prefer small, testable edits."
ACTOR_SYSTEM = """You are a coding agent that must output exactly one JSON object.
Choose one tool call at a time. Inspect files before editing. Run tests after editing.
Schema: {"tool": "tool_name", "arguments": {}, "reason": "brief reason"}"""


@dataclass(frozen=True)
class RunResult:
    run_id: str
    workspace: Path
    trace_path: Path
    outcome: str
    steps: int
    final_summary: str


def run_task(repo: Path, task: str, test_command: str, config: AgentConfig, model: ChatModel) -> RunResult:
    sandbox = Sandbox.create(repo, config.runs_dir)
    tools = ToolRegistry(sandbox, timeout=config.command_timeout)
    history: list[dict[str, Any]] = []
    write_event(
        sandbox.trace_path,
        "run_started",
        {"run_id": sandbox.run_id, "source_repo": str(sandbox.source_repo), "workspace": str(sandbox.workspace), "task": task},
    )

    plan = model.complete(PLANNER_SYSTEM, f"Create a concise plan for this task:\n{task}")
    write_event(sandbox.trace_path, "plan", {"plan": plan})

    final_summary = ""
    outcome = "incomplete"
    for step in range(1, config.max_steps + 1):
        raw = model.complete(
            ACTOR_SYSTEM,
            build_actor_prompt(task, test_command, plan, tools.descriptions(), history),
        )
        try:
            action = extract_json_object(raw)
            tool = str(action.get("tool", ""))
            arguments = action.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
        except Exception as exc:  # noqa: BLE001
            action = {"tool": "finish", "arguments": {"summary": f"Invalid model JSON: {exc}"}, "reason": "invalid_json"}
            tool = "finish"
            arguments = action["arguments"]

        if tool == "run_tests" and not arguments.get("command"):
            arguments["command"] = test_command
        result = tools.run(tool, arguments)
        record = {"tool": tool, "arguments": arguments, "reason": action.get("reason", ""), "result": result_to_dict(result)}
        history.append(record)
        write_event(sandbox.trace_path, "tool_call", {"step": step, **record})

        if tool == "finish":
            final_summary = str(arguments.get("summary", result.output))
            outcome = classify_outcome(history)
            break
    else:
        outcome = classify_outcome(history)
        final_summary = f"Stopped after max_steps={config.max_steps}."

    diff = tools.run("git_diff", {})
    write_event(sandbox.trace_path, "run_finished", {"outcome": outcome, "summary": final_summary, "diff": diff.output})
    return RunResult(
        run_id=sandbox.run_id,
        workspace=sandbox.workspace,
        trace_path=sandbox.trace_path,
        outcome=outcome,
        steps=len(history),
        final_summary=final_summary,
    )


def build_actor_prompt(task: str, test_command: str, plan: str, tools: str, history: list[dict[str, Any]]) -> str:
    compact_history = json.dumps(history[-8:], ensure_ascii=False, indent=2)
    return f"""Task:
{task}

Plan:
{plan}

Test command:
{test_command}

{tools}

Recent history:
{compact_history}

Choose the next single tool call. If complete, call finish."""


def result_to_dict(result: ToolResult) -> dict[str, Any]:
    return asdict(result)


def classify_outcome(history: list[dict[str, Any]]) -> str:
    if not history:
        return "no_actions"
    tests = [item for item in history if item["tool"] == "run_tests"]
    if tests and tests[-1]["result"]["ok"]:
        return "success"
    if any(item["result"].get("blocked") for item in history):
        return "blocked_by_policy"
    if any(item["result"].get("error_type") == "test_failed" for item in history):
        return "test_failed"
    if history[-1]["tool"] == "finish":
        return "finished_without_tests"
    return "incomplete"
