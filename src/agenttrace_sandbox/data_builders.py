from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenttrace_sandbox.tracing import read_jsonl


@dataclass(frozen=True)
class BuildSummary:
    count: int
    output_path: Path
    extra_path: Path | None = None

    def render(self) -> str:
        lines = [f"count={self.count}", f"output={self.output_path}"]
        if self.extra_path:
            lines.append(f"extra={self.extra_path}")
        return "\n".join(lines)


OFFLINE_BENCHMARK_TASKS = [
    {
        "id": "repair_subtract_operator",
        "function": "subtract",
        "description": "Fix subtract so it returns a minus b.",
        "buggy_code": "def subtract(a, b):\n    return a + b  # BUG\n",
        "tests": [
            "self.assertEqual(subtract(5, 3), 2)",
            "self.assertEqual(subtract(-1, -3), 2)",
        ],
    },
    {
        "id": "repair_is_even_logic",
        "function": "is_even",
        "description": "Fix is_even so it returns True only for even integers.",
        "buggy_code": "def is_even(n):\n    return n % 2 == 1  # BUG\n",
        "tests": [
            "self.assertTrue(is_even(4))",
            "self.assertFalse(is_even(5))",
        ],
    },
    {
        "id": "repair_reverse_string",
        "function": "reverse_string",
        "description": "Fix reverse_string so it returns the input string in reverse order.",
        "buggy_code": "def reverse_string(text):\n    return text  # BUG\n",
        "tests": [
            "self.assertEqual(reverse_string('abc'), 'cba')",
            "self.assertEqual(reverse_string(''), '')",
        ],
    },
    {
        "id": "repair_max_of_list",
        "function": "max_of_list",
        "description": "Fix max_of_list so it returns the largest number in a non-empty list.",
        "buggy_code": "def max_of_list(values):\n    return min(values)  # BUG\n",
        "tests": [
            "self.assertEqual(max_of_list([1, 4, 2]), 4)",
            "self.assertEqual(max_of_list([-5, -2, -9]), -2)",
        ],
    },
    {
        "id": "repair_count_vowels",
        "function": "count_vowels",
        "description": "Fix count_vowels so it counts lowercase and uppercase vowels.",
        "buggy_code": "def count_vowels(text):\n    return sum(1 for ch in text if ch in 'aeiou')  # BUG\n",
        "tests": [
            "self.assertEqual(count_vowels('AgentTrace'), 4)",
            "self.assertEqual(count_vowels('XYZ'), 0)",
        ],
    },
]


def build_benchmark_tasks(output_dir: Path, limit: int = 5, source_path: Path | None = None) -> BuildSummary:
    records = list(read_jsonl(source_path)) if source_path else OFFLINE_BENCHMARK_TASKS
    selected = records[:limit]
    repo_root = output_dir / "repos"
    manifest_path = output_dir / "tasks.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for raw in selected:
            task = normalize_benchmark_record(raw)
            repo_path = repo_root / task["id"]
            write_unit_repo(repo_path, task["function"], task["buggy_code"], task["tests"])
            manifest.write(
                json.dumps(
                    {
                        "id": task["id"],
                        "source": task.get("source", "offline_benchmark"),
                        "repo": f"repos/{task['id']}",
                        "task": task["description"],
                        "target_file": "solution.py",
                        "target_symbol": task["function"],
                        "test_command": "python3 -m unittest discover -s tests",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return BuildSummary(count=len(selected), output_path=manifest_path, extra_path=repo_root)


def normalize_benchmark_record(raw: dict[str, Any]) -> dict[str, Any]:
    required = ["id", "function", "description", "buggy_code", "tests"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"benchmark record missing fields: {', '.join(missing)}")
    tests = raw["tests"]
    if not isinstance(tests, list) or not all(isinstance(item, str) for item in tests):
        raise ValueError("benchmark record tests must be a list of strings")
    return {
        "id": safe_id(str(raw["id"])),
        "function": str(raw["function"]),
        "description": str(raw["description"]),
        "buggy_code": str(raw["buggy_code"]).rstrip() + "\n",
        "tests": tests,
        "source": str(raw.get("source", "offline_benchmark")),
    }


def write_unit_repo(repo_path: Path, function_name: str, buggy_code: str, tests: list[str]) -> None:
    repo_path.mkdir(parents=True, exist_ok=True)
    (repo_path / "solution.py").write_text(buggy_code, encoding="utf-8")
    test_lines = [
        "import unittest",
        "",
        f"from solution import {function_name}",
        "",
        "",
        "class GeneratedTests(unittest.TestCase):",
    ]
    for index, assertion in enumerate(tests, 1):
        test_lines.extend([f"    def test_case_{index}(self) -> None:", f"        {assertion}", ""])
    (repo_path / "tests").mkdir(exist_ok=True)
    (repo_path / "tests" / "test_solution.py").write_text("\n".join(test_lines).rstrip() + "\n", encoding="utf-8")
    (repo_path / "AGENT.md").write_text("Make the smallest code change. Run tests after editing.\n", encoding="utf-8")


def build_pr_wiki(input_path: Path, output_path: Path) -> BuildSummary:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for record in read_jsonl(input_path):
            wiki = make_pr_wiki(record)
            out.write(json.dumps(wiki, ensure_ascii=False) + "\n")
            count += 1
    return BuildSummary(count=count, output_path=output_path)


def make_pr_wiki(record: dict[str, Any]) -> dict[str, Any]:
    diff = str(record.get("diff", ""))
    files = record.get("files") or files_from_diff(diff)
    issue = compact_text(record.get("issue_body") or record.get("issue") or "")
    pr = compact_text(record.get("pr_body") or record.get("pr") or "")
    title = str(record.get("issue_title") or record.get("pr_title") or record.get("id") or "unknown")
    return {
        "id": record.get("id") or safe_id(title),
        "repo": record.get("repo", ""),
        "issue_title": record.get("issue_title", ""),
        "pr_title": record.get("pr_title", ""),
        "files": files,
        "wiki": {
            "bug_summary": summarize_text(title, issue),
            "change_summary": summarize_diff(diff),
            "fix_strategy": "Inspect the affected files, reproduce or reason about the failing behavior, make the smallest fix, then validate with tests.",
            "validation": record.get("test_command") or "Run the relevant unit tests or project test command.",
            "source_context": {"issue_excerpt": issue[:800], "pr_excerpt": pr[:800]},
        },
    }


def summarize_text(title: str, body: str) -> str:
    body = compact_text(body)
    if body:
        return f"{title}: {body[:300]}"
    return title


def summarize_diff(diff: str) -> str:
    files = files_from_diff(diff)
    added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    if files:
        return f"Patch touches {', '.join(files[:8])}; added lines={added}, removed lines={removed}."
    return "No unified diff was provided; use issue and PR text to infer the repair."


def files_from_diff(diff: str) -> list[str]:
    files: list[str] = []
    for match in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", diff, flags=re.MULTILINE):
        path = match.group(2)
        if path not in files:
            files.append(path)
    return files


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"
