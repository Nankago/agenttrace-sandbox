from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agenttrace_sandbox.config import AgentConfig
from agenttrace_sandbox.llm import ChatModel, extract_json_object
from agenttrace_sandbox.sandbox import Sandbox
from agenttrace_sandbox.tools import ToolRegistry, ToolResult
from agenttrace_sandbox.tracing import write_event


PLANNER_SYSTEM = """You are a careful coding-agent planner.
Use only the available file/test tools. Prefer the smallest code change that satisfies the task.
Do not propose commits, pull requests, package installs, or extra test rewrites unless the user explicitly asks.
"""
ACTOR_SYSTEM = """You are a coding agent that must output exactly one JSON object.
Output raw JSON only: no markdown, no code fences, no prose before or after the JSON.
Choose one tool call at a time. Inspect files before editing. Run tests after editing.
Prefer minimal code edits. Do not modify tests unless the task asks for tests or the existing tests are clearly wrong.
Schema: {"tool": "tool_name", "arguments": {}, "reason": "brief reason"}"""
REPAIR_SYSTEM = """You repair invalid tool-call responses.
Return raw JSON only, with no markdown or prose, matching:
{"tool": "tool_name", "arguments": {}, "reason": "brief reason"}"""


@dataclass(frozen=True)
class RunResult:
    run_id: str
    workspace: Path
    trace_path: Path
    outcome: str
    steps: int
    final_summary: str


@dataclass(frozen=True)
class StepDecision:
    raw: str
    action: dict[str, Any]
    parse_error: str
    retries_used: int
    format_violation: bool


def run_task(repo: Path, task: str, test_command: str, config: AgentConfig, model: ChatModel) -> RunResult:
    started_at = time.perf_counter()
    sandbox = Sandbox.create(
        repo,
        config.runs_dir,
        backend=config.sandbox_backend,
        docker_image=config.docker_image,
        docker_network=config.docker_network,
        docker_memory=config.docker_memory,
        docker_cpus=config.docker_cpus,
    )
    tools = ToolRegistry(sandbox, timeout=config.command_timeout)
    history: list[dict[str, Any]] = []
    write_event(
        sandbox.trace_path,
        "run_started",
        {
            "run_id": sandbox.run_id,
            "source_repo": str(sandbox.source_repo),
            "workspace": str(sandbox.workspace),
            "task": task,
            "provider": config.provider,
            "model": config.model,
            "temperature": config.temperature,
            "sandbox_backend": sandbox.backend,
            "docker_image": sandbox.docker_image if sandbox.backend == "docker" else "",
        },
    )

    plan = model.complete(PLANNER_SYSTEM, f"Create a concise plan for this task:\n{task}")
    write_event(sandbox.trace_path, "plan", {"plan": plan})

    final_summary = ""
    outcome = "incomplete"
    for step in range(1, config.max_steps + 1):
        decision = next_action(
            model=model,
            task=task,
            test_command=test_command,
            plan=plan,
            tools=tools.descriptions(),
            history=history,
            max_retries=config.json_retries,
        )
        record = execute_step(decision, tools, test_command)
        history.append(record)
        write_event(sandbox.trace_path, "tool_call", {"step": step, **record})

        if decision.parse_error:
            final_summary = str(decision.action.get("arguments", {}).get("summary") or f"Invalid model JSON: {decision.parse_error}")
            outcome = classify_outcome(history, max_steps_reached=False)
            break
        if record["tool"] == "finish":
            final_summary = str(record["arguments"].get("summary", record["result"]["output"]))
            outcome = classify_outcome(history, max_steps_reached=False)
            break
    else:
        outcome = classify_outcome(history, max_steps_reached=True)
        final_summary = f"Stopped after max_steps={config.max_steps}."

    diff = tools.run("git_diff", {})
    total_elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    write_event(
        sandbox.trace_path,
        "run_finished",
        {"outcome": outcome, "summary": final_summary, "diff": diff.output, "elapsed_ms": total_elapsed_ms},
    )
    return RunResult(
        run_id=sandbox.run_id,
        workspace=sandbox.workspace,
        trace_path=sandbox.trace_path,
        outcome=outcome,
        steps=len(history),
        final_summary=final_summary,
    )


def next_action(
    model: ChatModel,
    task: str,
    test_command: str,
    plan: str,
    tools: str,
    history: list[dict[str, Any]],
    max_retries: int,
) -> StepDecision:
    prompt = build_actor_prompt(task, test_command, plan, tools, history)
    raw = model.complete(ACTOR_SYSTEM, prompt)
    for retry in range(max_retries + 1):
        try:
            return StepDecision(
                raw=raw,
                action=normalize_action(extract_json_object(raw)),
                parse_error="",
                retries_used=retry,
                format_violation=not is_raw_json_object(raw),
            )
        except Exception as exc:  # noqa: BLE001
            if retry >= max_retries:
                action = {
                    "tool": "finish",
                    "arguments": {"summary": f"Invalid model JSON after {max_retries + 1} attempt(s): {exc}"},
                    "reason": "invalid_json",
                }
                return StepDecision(raw=raw, action=action, parse_error=str(exc), retries_used=retry, format_violation=True)
            raw = model.complete(
                REPAIR_SYSTEM,
                f"Original response was not valid JSON for the tool schema.\nError: {exc}\nResponse:\n{raw}",
            )
    return StepDecision(
        raw=raw,
        action={"tool": "finish", "arguments": {"summary": "unreachable"}, "reason": "invalid_json"},
        parse_error="unreachable",
        retries_used=max_retries,
        format_violation=True,
    )


def execute_step(decision: StepDecision, tools: ToolRegistry, test_command: str) -> dict[str, Any]:
    started_at = time.perf_counter()
    tool = str(decision.action.get("tool", ""))
    arguments = decision.action.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
    if tool == "run_tests" and not arguments.get("command"):
        arguments["command"] = test_command

    result = tools.run(tool, arguments)
    if decision.parse_error:
        result = ToolResult(False, decision.parse_error, error_type="invalid_json")
    return {
        "tool": tool,
        "arguments": arguments,
        "reason": decision.action.get("reason", ""),
        "raw_output": decision.raw,
        "parse_error": decision.parse_error,
        "retries_used": decision.retries_used,
        "format_violation": decision.format_violation,
        "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "result": result_to_dict(result),
    }


def normalize_action(action: dict[str, Any]) -> dict[str, Any]:
    tool = action.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        raise ValueError("tool must be a non-empty string")
    arguments = action.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    reason = action.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    return {"tool": tool, "arguments": arguments, "reason": reason}


def is_raw_json_object(text: str) -> bool:
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False
    try:
        return isinstance(json.loads(stripped), dict)
    except json.JSONDecodeError:
        return False


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


def classify_outcome(history: list[dict[str, Any]], max_steps_reached: bool = False) -> str:
    if not history:
        return "no_actions"
    if any(item["result"].get("error_type") == "invalid_json" for item in history):
        return "invalid_json"
    tests = [item for item in history if item["tool"] == "run_tests"]
    if tests and tests[-1]["result"]["ok"]:
        return "success"
    if any(item["result"].get("blocked") for item in history):
        return "blocked_by_policy"
    if any(item["result"].get("error_type") == "edit_miss" for item in history):
        return "edit_miss"
    if any(item["result"].get("error_type") == "test_failed" for item in history):
        return "test_failed"
    if max_steps_reached:
        return "max_steps"
    if history[-1]["tool"] == "finish":
        return "finished_without_tests"
    return "incomplete"
