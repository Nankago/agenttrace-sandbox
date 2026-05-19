from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Any

from agenttrace_sandbox.tracing import read_jsonl

SYSTEM_PROMPT = "You are a coding agent. Return exactly one safe JSON tool call for the next step."
SFT_INSTRUCTION = "Given a coding task, plan, and previous tool history, choose the next safe tool call as JSON."
REPAIR_SFT_TASKS = {"localize_files", "explain_bug", "repair_rationale", "test_spec", "repair_instruction"}
REPAIR_SFT_VARIANTS = {"full", "no-tests", "no-llm", "diff-only"}
BOILERPLATE_POLICIES = {"keep", "light", "strict", "semantic"}


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
    variant: str = "full",
    boilerplate_policy: str = "semantic",
) -> int:
    if variant not in REPAIR_SFT_VARIANTS:
        raise ValueError(f"unknown repair SFT variant: {variant}")
    if boilerplate_policy not in BOILERPLATE_POLICIES:
        raise ValueError(f"unknown boilerplate policy: {boilerplate_policy}")
    selected_tasks = tasks or sorted(REPAIR_SFT_TASKS)
    unknown = [task for task in selected_tasks if task not in REPAIR_SFT_TASKS]
    if unknown:
        raise ValueError(f"unknown repair SFT task(s): {', '.join(unknown)}")
    samples = collect_repair_sft_samples(
        input_path,
        selected_tasks,
        min_quality=min_quality,
        require_grounding=require_grounding,
        variant=variant,
        boilerplate_policy=boilerplate_policy,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(samples, output_path)
    return len(samples)


def export_repair_corpus(
    input_path: Path,
    output_path: Path,
    min_quality: float = 0.0,
    require_grounding: bool = False,
    output_format: str = "jsonl",
    max_evidence_chars: int = 1200,
    include_raw_diff: bool = False,
    boilerplate_policy: str = "semantic",
) -> int:
    if output_format != "jsonl":
        raise ValueError(f"unsupported repair corpus format: {output_format}")
    if boilerplate_policy not in BOILERPLATE_POLICIES:
        raise ValueError(f"unknown boilerplate policy: {boilerplate_policy}")
    rows = collect_repair_corpus_records(
        input_path,
        min_quality=min_quality,
        require_grounding=require_grounding,
        max_evidence_chars=max_evidence_chars,
        include_raw_diff=include_raw_diff,
        boilerplate_policy=boilerplate_policy,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, output_path)
    return len(rows)


def collect_repair_sft_samples(
    input_path: Path,
    tasks: list[str],
    min_quality: float = 0.0,
    require_grounding: bool = False,
    variant: str = "full",
    boilerplate_policy: str = "semantic",
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for card in read_jsonl(input_path):
        quality = float(card.get("quality", {}).get("overall", 0))
        if quality < min_quality:
            continue
        for task in tasks:
            sample = make_repair_sft_sample(card, task, require_grounding=require_grounding, variant=variant, boilerplate_policy=boilerplate_policy)
            if sample:
                samples.append(sample)
    return samples


def make_repair_sft_sample(
    card: dict[str, Any],
    task: str,
    require_grounding: bool = False,
    variant: str = "full",
    boilerplate_policy: str = "semantic",
) -> dict[str, Any] | None:
    evidence = variant_evidence(card.get("evidence", []), variant)
    evidence_text = format_repair_evidence(evidence, boilerplate_policy=boilerplate_policy)
    source_id = str(card.get("id", ""))
    repo = str(card.get("repo", ""))
    quality = float(card.get("quality", {}).get("overall", 0))
    llm_card = card.get("llm_repair_card", {}) if variant == "full" and isinstance(card.get("llm_repair_card"), dict) else {}
    repair_card = card.get("repair_card", {}) if isinstance(card.get("repair_card"), dict) else {}

    if task == "localize_files":
        localization = repair_card.get("localization", {}) if isinstance(repair_card.get("localization"), dict) else {}
        output = {
            "source_files": localization.get("source_files", card.get("source_files", [])),
            "test_files": [] if variant in {"no-tests", "diff-only"} else localization.get("test_files", card.get("test_files", [])),
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
        if variant == "diff-only":
            output, evidence_ids = diff_only_bug_summary(evidence)
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
        if variant in {"no-tests", "diff-only"}:
            return None
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
        output = build_repair_instruction(card, repair_card, llm_card, variant=variant)
        evidence_ids = repair_instruction_evidence_ids(repair_card, llm_card)
        instruction = "Write a concise repair instruction for a coding agent."
    else:
        return None

    if not output:
        return None
    valid_ids = {str(item.get("id")) for item in evidence if item.get("id")}
    evidence_ids = [str(item) for item in evidence_ids if str(item)]
    if variant in {"no-tests", "diff-only"}:
        evidence_ids = [item for item in evidence_ids if item in valid_ids]
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
            "variant": variant,
        },
    }


def collect_repair_corpus_records(
    input_path: Path,
    min_quality: float = 0.0,
    require_grounding: bool = False,
    max_evidence_chars: int = 1200,
    include_raw_diff: bool = False,
    boilerplate_policy: str = "light",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for card in read_jsonl(input_path):
        quality = float(card.get("quality", {}).get("overall", 0))
        llm_quality = card.get("llm_quality", {}) if isinstance(card.get("llm_quality"), dict) else {}
        grounding_ok = bool(llm_quality.get("grounding_ok"))
        if quality < min_quality:
            continue
        if require_grounding and not grounding_ok:
            continue
        rows.append(
            make_repair_corpus_record(
                card,
                max_evidence_chars=max_evidence_chars,
                include_raw_diff=include_raw_diff,
                boilerplate_policy=boilerplate_policy,
            )
        )
    return rows


def make_repair_corpus_record(
    card: dict[str, Any],
    max_evidence_chars: int = 1200,
    include_raw_diff: bool = False,
    boilerplate_policy: str = "light",
) -> dict[str, Any]:
    quality = card.get("quality", {}) if isinstance(card.get("quality"), dict) else {}
    llm_quality = card.get("llm_quality", {}) if isinstance(card.get("llm_quality"), dict) else {}
    return {
        "id": str(card.get("id", "")),
        "repo": str(card.get("repo", "")),
        "text": repair_corpus_text(card, max_evidence_chars=max_evidence_chars, include_raw_diff=include_raw_diff, boilerplate_policy=boilerplate_policy),
        "metadata": {
            "quality": quality.get("overall", 0),
            "bug_fix_score": quality.get("bug_fix_score", 0),
            "has_test_evidence": bool(quality.get("has_test_evidence")),
            "grounding_ok": bool(llm_quality.get("grounding_ok")),
        },
    }


def repair_corpus_text(
    card: dict[str, Any],
    max_evidence_chars: int = 1200,
    include_raw_diff: bool = False,
    boilerplate_policy: str = "light",
) -> str:
    repair_card = card.get("repair_card", {}) if isinstance(card.get("repair_card"), dict) else {}
    llm_card = card.get("llm_repair_card", {}) if isinstance(card.get("llm_repair_card"), dict) else {}
    source_record = card.get("source_record", {}) if isinstance(card.get("source_record"), dict) else {}
    quality = card.get("quality", {}) if isinstance(card.get("quality"), dict) else {}
    llm_quality = card.get("llm_quality", {}) if isinstance(card.get("llm_quality"), dict) else {}

    symptom = clean_repair_text(repair_claim_text(repair_card, "symptom"), boilerplate_policy)
    patch_intent = repair_claim_text(repair_card, "patch_intent")
    test_oracle = repair_claim_text(repair_card, "test_oracle")
    validation = repair_claim_text(repair_card, "validation")
    root_cause = claim_text(llm_card.get("root_cause")) or symptom
    failure_condition = claim_text(llm_card.get("failure_condition")) or symptom
    expected_behavior = claim_text(llm_card.get("expected_behavior")) or test_oracle
    rationale = claim_text(llm_card.get("repair_rationale")) or patch_intent
    edge_cases = [claim_text(item) for item in llm_card.get("edge_cases", []) if claim_text(item)] if isinstance(llm_card.get("edge_cases"), list) else []

    parts = [
        f"Repository: {card.get('repo', '')}",
        f"PR: {source_record.get('pr_number', '')} {source_record.get('pr_url', '')}".rstrip(),
        f"Issue: {source_record.get('issue_number', '')}".rstrip(),
        f"Problem Summary: {symptom}",
        "Evidence:\n"
        + corpus_evidence_text(
            card.get("evidence", []),
            max_evidence_chars=max_evidence_chars,
            include_raw_diff=include_raw_diff,
            boilerplate_policy=boilerplate_policy,
        ),
        "Changed source files: " + join_values(card.get("source_files", [])),
        "Changed test files: " + join_values(card.get("test_files", [])),
        f"Symptom: {symptom}",
        f"Root Cause: {root_cause}",
        f"Failure Condition: {failure_condition}",
        f"Expected Behavior: {expected_behavior}",
        f"Patch Intent / Repair Rationale: {rationale}",
        f"Test Oracle: {test_oracle}",
        "Edge Cases: " + (join_values(edge_cases) if edge_cases else "insufficient_evidence"),
        "Validation / Quality signals: "
        + join_values(
            [
                f"quality.overall={quality.get('overall', 0)}",
                f"bug_fix_score={quality.get('bug_fix_score', 0)}",
                f"has_test_evidence={bool(quality.get('has_test_evidence'))}",
                f"has_source_patch={bool(quality.get('has_source_patch'))}",
                f"llm_grounding_ok={bool(llm_quality.get('grounding_ok'))}",
                validation,
            ]
        ),
    ]
    return redact_secrets("\n\n".join(part for part in parts if part.strip()))


def corpus_evidence_text(evidence: Any, max_evidence_chars: int, include_raw_diff: bool, boilerplate_policy: str = "light") -> str:
    if not isinstance(evidence, list):
        return ""
    chunks: list[str] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", ""))
        if item.get("type") in {"source_diff", "test_diff", "other_diff"} and not include_raw_diff:
            text = summarize_evidence_diff(text)
        else:
            text = clean_repair_text(text, boilerplate_policy)[:max_evidence_chars]
        file = f" {item.get('file')}" if item.get("file") else ""
        chunks.append(f"- [{item.get('id')}: {item.get('type')}{file}] {text}")
    return "\n".join(chunks)


def stats_repair_cards(input_pattern: Path | str) -> dict[str, Any]:
    paths = repair_card_paths(input_pattern)
    records = [record for path in paths for record in read_jsonl(path)]
    qualities = [float(record.get("quality", {}).get("overall", 0)) for record in records]
    bug_scores = [float(record.get("quality", {}).get("bug_fix_score", 0)) for record in records]
    result: dict[str, Any] = {
        "files": [str(path) for path in paths],
        "records": len(records),
        "quality": {
            "avg": round(sum(qualities) / len(qualities), 3) if qualities else 0.0,
            "min": min(qualities) if qualities else 0.0,
            "max": max(qualities) if qualities else 0.0,
        },
        "has_test_evidence": count_ratio(records, lambda record: bool(record.get("quality", {}).get("has_test_evidence"))),
        "has_source_patch": count_ratio(records, lambda record: bool(record.get("quality", {}).get("has_source_patch"))),
        "avg_bug_fix_score": round(sum(bug_scores) / len(bug_scores), 3) if bug_scores else 0.0,
        "derived_tasks": {},
    }
    for field in ("docs_only", "tests_only"):
        if any(field in record for record in records):
            result[field] = count_ratio(records, lambda record, key=field: bool(record.get(key)))
    llm_records = [record for record in records if isinstance(record.get("llm_quality"), dict)]
    if llm_records:
        coverages = [float(record.get("llm_quality", {}).get("field_coverage", 0)) for record in llm_records]
        result["llm_quality"] = {
            "valid_json": count_ratio(llm_records, lambda record: bool(record.get("llm_quality", {}).get("valid_json"))),
            "grounding_ok": count_ratio(llm_records, lambda record: bool(record.get("llm_quality", {}).get("grounding_ok"))),
            "avg_field_coverage": round(sum(coverages) / len(coverages), 3) if coverages else 0.0,
        }
    task_counts: dict[str, int] = {}
    for record in records:
        derived = record.get("derived_tasks", {})
        if isinstance(derived, dict):
            for task in derived:
                task_counts[str(task)] = task_counts.get(str(task), 0) + 1
    result["derived_tasks"] = dict(sorted(task_counts.items()))
    return result


def render_repair_card_stats(stats: dict[str, Any]) -> str:
    lines = [
        f"files={len(stats.get('files', []))}",
        f"records={stats.get('records', 0)}",
        "quality.overall="
        f"avg={stats.get('quality', {}).get('avg', 0)} "
        f"min={stats.get('quality', {}).get('min', 0)} "
        f"max={stats.get('quality', {}).get('max', 0)}",
        render_count_ratio("has_test_evidence", stats.get("has_test_evidence", {})),
        render_count_ratio("has_source_patch", stats.get("has_source_patch", {})),
        f"avg_bug_fix_score={stats.get('avg_bug_fix_score', 0)}",
    ]
    for key in ("docs_only", "tests_only"):
        if key in stats:
            lines.append(render_count_ratio(key, stats.get(key, {})))
    if "llm_quality" in stats:
        llm = stats["llm_quality"]
        lines.extend(
            [
                render_count_ratio("llm.valid_json", llm.get("valid_json", {})),
                render_count_ratio("llm.grounding_ok", llm.get("grounding_ok", {})),
                f"llm.avg_field_coverage={llm.get('avg_field_coverage', 0)}",
            ]
        )
    derived = stats.get("derived_tasks", {})
    if derived:
        lines.append("derived_tasks=" + ", ".join(f"{task}:{count}" for task, count in derived.items()))
    return "\n".join(lines)


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


def format_repair_evidence(evidence: Any, boilerplate_policy: str = "light") -> str:
    if not isinstance(evidence, list):
        return ""
    chunks: list[str] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        evidence_id = item.get("id", "")
        source_type = item.get("type", "")
        file = item.get("file", "")
        text = clean_repair_text(str(item.get("text", "")), boilerplate_policy)
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


def variant_evidence(evidence: Any, variant: str) -> list[dict[str, Any]]:
    if not isinstance(evidence, list):
        return []
    allowed = None
    if variant == "diff-only":
        allowed = {"pr_text", "source_diff", "other_diff"}
    elif variant == "no-tests":
        allowed = {"issue_text", "pr_text", "source_diff", "other_diff"}
    rows: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        if allowed is not None and item.get("type") not in allowed:
            continue
        rows.append(item)
    return rows


def build_repair_instruction(card: dict[str, Any], repair_card: dict[str, Any], llm_card: dict[str, Any], variant: str = "full") -> str:
    source_files = card.get("source_files", [])
    if not source_files and isinstance(repair_card.get("localization"), dict):
        source_files = repair_card["localization"].get("source_files", [])
    target = join_values(source_files[:5]) if isinstance(source_files, list) and source_files else "the affected source files"
    root_cause = claim_text(llm_card.get("root_cause"))
    failure = claim_text(llm_card.get("failure_condition"))
    expected = claim_text(llm_card.get("expected_behavior"))
    rationale = claim_text(llm_card.get("repair_rationale"))

    if variant == "diff-only":
        summary, _ids = diff_only_bug_summary(variant_evidence(card.get("evidence", []), variant))
        return f"Update {target} according to the PR/source diff evidence. Preserve the intended source behavior: {summary}".strip()

    details = [text for text in [root_cause, failure, expected, rationale] if text]
    if details:
        first = details[0]
        rest = " ".join(details[1:3])
        suffix = f" {rest}" if rest else ""
        return f"Fix {target}: {first}.{suffix}".strip()

    derived = card.get("derived_tasks", {}) if isinstance(card.get("derived_tasks"), dict) else {}
    repair_instruction = derived.get("repair_instruction", {}) if isinstance(derived.get("repair_instruction"), dict) else {}
    return str(repair_instruction.get("output", ""))


def repair_instruction_evidence_ids(repair_card: dict[str, Any], llm_card: dict[str, Any]) -> list[str]:
    claims = [
        llm_card.get("root_cause"),
        llm_card.get("failure_condition"),
        llm_card.get("expected_behavior"),
        llm_card.get("repair_rationale"),
    ]
    ids = claim_evidence_ids([claim for claim in claims if isinstance(claim, dict)])
    if ids:
        return ids
    localization = repair_card.get("localization", {}) if isinstance(repair_card.get("localization"), dict) else {}
    values = localization.get("evidence_ids", [])
    return list(values) if isinstance(values, list) else []


def diff_only_bug_summary(evidence: list[dict[str, Any]]) -> tuple[str, list[str]]:
    ids: list[str] = []
    parts: list[str] = []
    for item in evidence:
        evidence_id = str(item.get("id", ""))
        if evidence_id:
            ids.append(evidence_id)
        if item.get("type") == "pr_text":
            text = str(item.get("text", "")).strip()
            if text:
                parts.append(text[:300])
        elif item.get("type") in {"source_diff", "other_diff"}:
            file = str(item.get("file", ""))
            parts.append(f"{file}: {summarize_evidence_diff(str(item.get('text', '')))}".strip())
    return (" ".join(parts).strip() or "Use the PR and source diff evidence to infer the repair.", ids)


def repair_claim_text(container: dict[str, Any], key: str) -> str:
    value = container.get(key)
    return claim_text(value)


def claim_text(value: Any) -> str:
    if usable_claim(value):
        return str(value.get("text", ""))
    return ""


def summarize_evidence_diff(diff: str) -> str:
    if is_binary_or_low_signal_diff(diff):
        return "low-signal or binary diff omitted"
    added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    files = []
    for line in diff.splitlines():
        match = re.match(r"^diff --git a/(.*?) b/(.*?)$", line)
        if match:
            files.append(match.group(2))
    file_text = f"files={join_values(files)}; " if files else ""
    hunks = sum(1 for line in diff.splitlines() if line.startswith("@@"))
    return f"{file_text}hunks={hunks}; added lines={added}, removed lines={removed}"


def clean_repair_text(text: str, policy: str = "light") -> str:
    if policy == "keep" or not text:
        return text
    cleaned = remove_html_comments(text)
    cleaned = remove_template_sections(cleaned, ["AI Assistance Disclosure", "Checklist", "Backport"], keep_semantic_tail=True)
    cleaned = remove_checkbox_boilerplate(cleaned)
    if policy in {"strict", "semantic"}:
        cleaned = remove_template_sentences(cleaned)
    if policy == "semantic":
        cleaned = keep_semantic_pr_sections(cleaned)
    return compact_spaces(cleaned)


def remove_html_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)


def remove_template_sections(text: str, headings: list[str], keep_semantic_tail: bool = True) -> str:
    semantic_next = (
        r"(?:Trac ticket number|Branch description|Problem|Description|Reproduction|Steps to reproduce|"
        r"Expected behavior|Actual behavior|Test plan|Tests|Validation|Summary)"
    )
    cleaned = text
    for heading in headings:
        pattern = rf"(?:^|\s)(?:#+\s*)?{re.escape(heading)}(?:\s*\([^)]*\))?.*?(?=(?:\s#+\s*{semantic_next}\b)|$)"
        if keep_semantic_tail:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
        else:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned


def remove_checkbox_boilerplate(text: str) -> str:
    lines = []
    noisy_fragments = [
        "contribution guidelines",
        "does not disclose a security vulnerability",
        "targets the `main` branch",
        "targets the main branch",
        "commit message is written in past tense",
        "please select exactly one",
        "no ai tools were used",
        "if ai tools were used",
        "fully reviewed and verified their output",
        "automated ai review",
        "has patch",
        "attached screenshots",
        "added or updated relevant docs",
    ]
    for line in text.splitlines():
        lowered = line.lower()
        if re.match(r"\s*[-*]\s*\[[ xX]\]", line) and any(fragment in lowered for fragment in noisy_fragments):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    for fragment in noisy_fragments:
        cleaned = re.sub(rf"[-*]?\s*\[[ xX]\]?\s*[^.]*{re.escape(fragment)}[^.]*\.?", " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def remove_template_sentences(text: str) -> str:
    patterns = [
        r"Backports will be evaluated and done by mergers[^.]*\.?",
        r"see vulnerability reporting[^.]*\.?",
        r"This PR follows the contribution guidelines[^.]*\.?",
        r"This PR targets the `?main`? branch[^.]*\.?",
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def keep_semantic_pr_sections(text: str) -> str:
    sections = split_markdown_sections(text)
    if not sections:
        return text
    semantic_headings = {
        "",
        "title",
        "summary",
        "description",
        "problem",
        "branch description",
        "reproduction",
        "steps to reproduce",
        "actual behavior",
        "expected behavior",
        "test plan",
        "tests",
        "tests run",
        "validation",
    }
    kept = []
    for heading, body in sections:
        normalized = normalize_heading(heading)
        if normalized in semantic_headings:
            kept.append(f"{heading} {body}".strip() if heading else body)
    return "\n".join(kept) if kept else text


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    heading_pattern = (
        r"Trac ticket number|Branch description|Steps to reproduce|Expected behavior|"
        r"Actual behavior|Test plan|Tests run|AI Assistance Disclosure(?:\s*\([^)]*\))?|"
        r"Checklist|Backport|Validation|Description|Reproduction|Problem|Summary|Tests"
    )
    matches = list(re.finditer(rf"(?:^|\s)(#{{2,6}})\s+({heading_pattern})\b", text, flags=re.IGNORECASE))
    if not matches:
        return []
    sections: list[tuple[str, str]] = []
    prefix = text[: matches[0].start()].strip()
    if prefix:
        sections.append(("", prefix))
    for index, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        heading, inline_body = split_known_semantic_heading(heading)
        if inline_body:
            body = f"{inline_body} {body}".strip()
        sections.append((heading, body))
    return sections


def split_known_semantic_heading(raw_heading: str) -> tuple[str, str]:
    lowered = compact_spaces(raw_heading).lower()
    known = [
        "trac ticket number",
        "branch description",
        "steps to reproduce",
        "expected behavior",
        "actual behavior",
        "test plan",
        "tests run",
        "validation",
        "description",
        "reproduction",
        "problem",
        "summary",
        "tests",
    ]
    for heading in known:
        if lowered == heading:
            return raw_heading, ""
        if lowered.startswith(heading + " "):
            return heading.title(), raw_heading[len(heading) :].strip(" :-")
    return raw_heading, ""


def normalize_heading(heading: str) -> str:
    heading = re.sub(r"\([^)]*\)", "", heading)
    heading = re.sub(r"[^A-Za-z0-9 ]+", " ", heading)
    return compact_spaces(heading).lower()


def is_binary_or_low_signal_diff(diff: str) -> bool:
    lowered = diff.lower()
    if "binary files " in lowered or "git binary patch" in lowered:
        return True
    files = []
    for line in diff.splitlines():
        match = re.match(r"^diff --git a/(.*?) b/(.*?)$", line)
        if match:
            files.append(match.group(2).lower())
    low_signal_suffixes = (
        ".lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "go.sum",
        ".min.js",
        ".min.css",
    )
    low_signal_parts = ("/vendor/", "/dist/", "/build/", "/generated/", "/migrations/")
    return bool(files) and all(path.endswith(low_signal_suffixes) or any(part in f"/{path}" for part in low_signal_parts) for path in files)


def compact_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def repair_card_paths(input_pattern: Path | str) -> list[Path]:
    pattern = str(input_pattern)
    paths = [Path(path) for path in glob.glob(pattern)] if any(char in pattern for char in "*?[]") else [Path(pattern)]
    return sorted(paths)


def count_ratio(records: list[dict[str, Any]], predicate: Any) -> dict[str, Any]:
    count = sum(1 for record in records if predicate(record))
    total = len(records)
    return {"count": count, "ratio": round(count / total, 3) if total else 0.0}


def render_count_ratio(name: str, value: dict[str, Any]) -> str:
    return f"{name}={value.get('count', 0)} ratio={value.get('ratio', 0)}"


def join_values(values: Any) -> str:
    if not isinstance(values, list):
        return str(values) if values else ""
    return ", ".join(str(value) for value in values if str(value))


def redact_secrets(text: str) -> str:
    patterns = [
        r"sk-[A-Za-z0-9_-]{12,}",
        r"(?:ghp|github_pat|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{12,}",
        r"Bearer\s+[A-Za-z0-9._-]{16,}",
    ]
    redacted = text
    for pattern in patterns:
        redacted = re.sub(pattern, "[REDACTED]", redacted)
    return redacted


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
