from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenttrace_sandbox.tracing import read_jsonl

SYSTEM_PROMPT = "You are a coding agent. Return exactly one safe JSON tool call for the next step."
SFT_INSTRUCTION = "Given a coding task, plan, and previous tool history, choose the next safe tool call as JSON."
REPAIR_SFT_TASKS = {"localize_files", "explain_bug", "repair_rationale", "test_spec", "repair_instruction"}


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


def export_repair_sft(
    input_path: Path,
    output_path: Path,
    tasks: list[str] | None = None,
    min_quality: float = 0.0,
    require_grounding: bool = False,
) -> int:
    selected_tasks = tasks or sorted(REPAIR_SFT_TASKS)
    unknown = [task for task in selected_tasks if task not in REPAIR_SFT_TASKS]
    if unknown:
        raise ValueError(f"unknown repair SFT task(s): {', '.join(unknown)}")
    samples = collect_repair_sft_samples(input_path, selected_tasks, min_quality=min_quality, require_grounding=require_grounding)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(samples, output_path)
    return len(samples)


def collect_repair_sft_samples(
    input_path: Path,
    tasks: list[str],
    min_quality: float = 0.0,
    require_grounding: bool = False,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for card in read_jsonl(input_path):
        quality = float(card.get("quality", {}).get("overall", 0))
        if quality < min_quality:
            continue
        for task in tasks:
            sample = make_repair_sft_sample(card, task, require_grounding=require_grounding)
            if sample:
                samples.append(sample)
    return samples


def make_repair_sft_sample(card: dict[str, Any], task: str, require_grounding: bool = False) -> dict[str, Any] | None:
    evidence = card.get("evidence", [])
    evidence_text = format_repair_evidence(evidence)
    source_id = str(card.get("id", ""))
    repo = str(card.get("repo", ""))
    quality = float(card.get("quality", {}).get("overall", 0))
    llm_card = card.get("llm_repair_card", {}) if isinstance(card.get("llm_repair_card"), dict) else {}
    repair_card = card.get("repair_card", {}) if isinstance(card.get("repair_card"), dict) else {}

    if task == "localize_files":
        localization = repair_card.get("localization", {}) if isinstance(repair_card.get("localization"), dict) else {}
        output = {
            "source_files": localization.get("source_files", card.get("source_files", [])),
            "test_files": localization.get("test_files", card.get("test_files", [])),
        }
        evidence_ids = list(localization.get("evidence_ids", [])) if isinstance(localization.get("evidence_ids", []), list) else []
        instruction = "Identify the source and test files involved in fixing the bug."
    elif task == "explain_bug":
        claims = [llm_card.get("failure_condition"), llm_card.get("root_cause"), llm_card.get("expected_behavior")]
        output = "\n".join(format_claim(label, claim) for label, claim in zip(["Failure condition", "Root cause", "Expected behavior"], claims) if usable_claim(claim))
        if not output:
            symptom = repair_card.get("symptom", {}) if isinstance(repair_card.get("symptom"), dict) else {}
            output = str(symptom.get("text", ""))
            evidence_ids = list(symptom.get("evidence_ids", [])) if isinstance(symptom.get("evidence_ids", []), list) else []
        else:
            evidence_ids = claim_evidence_ids([claim for claim in claims if isinstance(claim, dict)])
        instruction = "Explain the bug using only the provided evidence."
    elif task == "repair_rationale":
        claim = llm_card.get("repair_rationale")
        if usable_claim(claim):
            output = str(claim["text"])
            evidence_ids = list(claim.get("evidence_ids", []))
        else:
            patch_intent = repair_card.get("patch_intent", {}) if isinstance(repair_card.get("patch_intent"), dict) else {}
            output = str(patch_intent.get("text", ""))
            evidence_ids = list(patch_intent.get("evidence_ids", [])) if isinstance(patch_intent.get("evidence_ids", []), list) else []
        instruction = "Describe why the patch fixes the bug."
    elif task == "test_spec":
        expected = llm_card.get("expected_behavior")
        edge_cases = llm_card.get("edge_cases", []) if isinstance(llm_card.get("edge_cases"), list) else []
        parts = []
        if usable_claim(expected):
            parts.append(f"Expected behavior: {expected['text']}")
        parts.extend(f"Edge case: {case['text']}" for case in edge_cases if usable_claim(case))
        output = "\n".join(parts)
        evidence_ids = claim_evidence_ids(([expected] if isinstance(expected, dict) else []) + [case for case in edge_cases if isinstance(case, dict)])
        if not output:
            test_oracle = repair_card.get("test_oracle", {}) if isinstance(repair_card.get("test_oracle"), dict) else {}
            output = str(test_oracle.get("text", ""))
            evidence_ids = list(test_oracle.get("evidence_ids", [])) if isinstance(test_oracle.get("evidence_ids", []), list) else []
        instruction = "Summarize the expected behavior and edge cases from the test evidence."
    elif task == "repair_instruction":
        derived = card.get("derived_tasks", {}) if isinstance(card.get("derived_tasks"), dict) else {}
        repair_instruction = derived.get("repair_instruction", {}) if isinstance(derived.get("repair_instruction"), dict) else {}
        output = str(repair_instruction.get("output", ""))
        localization = repair_card.get("localization", {}) if isinstance(repair_card.get("localization"), dict) else {}
        evidence_ids = list(localization.get("evidence_ids", [])) if isinstance(localization.get("evidence_ids", []), list) else []
        instruction = "Write a concise repair instruction for a coding agent."
    else:
        return None

    if not output:
        return None
    valid_ids = {str(item.get("id")) for item in evidence if item.get("id")}
    evidence_ids = [str(item) for item in evidence_ids if str(item)]
    grounding_ok = bool(evidence_ids) and all(item in valid_ids for item in evidence_ids)
    if require_grounding and not grounding_ok:
        return None
    return {
        "instruction": instruction,
        "input": {
            "repo": repo,
            "source_id": source_id,
            "evidence": evidence_text,
        },
        "output": output,
        "metadata": {
            "task_type": task,
            "source_id": source_id,
            "repo": repo,
            "quality": quality,
            "evidence_ids": evidence_ids,
            "grounding_ok": grounding_ok,
        },
    }


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


def format_repair_evidence(evidence: Any) -> str:
    if not isinstance(evidence, list):
        return ""
    chunks: list[str] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        evidence_id = item.get("id", "")
        source_type = item.get("type", "")
        file = item.get("file", "")
        text = item.get("text", "")
        header = f"[{evidence_id}: {source_type}" + (f" {file}" if file else "") + "]"
        chunks.append(f"{header}\n{text}")
    return "\n\n".join(chunks)


def usable_claim(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get("text")) and value.get("text") != "insufficient_evidence"


def format_claim(label: str, claim: Any) -> str:
    if not usable_claim(claim):
        return ""
    return f"{label}: {claim['text']}"


def claim_evidence_ids(claims: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for claim in claims:
        values = claim.get("evidence_ids", [])
        if not isinstance(values, list):
            continue
        for value in values:
            evidence_id = str(value)
            if evidence_id and evidence_id not in ids:
                ids.append(evidence_id)
    return ids


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
