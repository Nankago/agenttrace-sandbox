from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
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

OFFLINE_MBPP_ROWS = [
    {
        "task_id": "offline_subtract",
        "text": "Write a function to subtract two numbers.",
        "code": "def subtract(a, b):\n    return a - b\n",
        "test_list": ["assert subtract(5, 3) == 2", "assert subtract(-1, -3) == 2"],
    },
    {
        "task_id": "offline_square",
        "text": "Write a function to return the square of a number.",
        "code": "def square(n):\n    return n * n\n",
        "test_list": ["assert square(4) == 16", "assert square(-3) == 9"],
    },
]

OFFLINE_HUMANEVAL_ROWS = [
    {
        "task_id": "HumanEval/offline_add",
        "prompt": 'def add(a: int, b: int) -> int:\n    """Return a plus b."""\n',
        "canonical_solution": "    return a + b\n",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n",
        "entry_point": "add",
    },
    {
        "task_id": "HumanEval/offline_is_palindrome",
        "prompt": 'def is_palindrome(text: str) -> bool:\n    """Return True if text reads the same forwards and backwards."""\n',
        "canonical_solution": "    return text == text[::-1]\n",
        "test": "def check(candidate):\n    assert candidate('level') is True\n    assert candidate('agent') is False\n",
        "entry_point": "is_palindrome",
    },
]


HF_MBPP_DATASET = "google-research-datasets/mbpp"
HF_HUMANEVAL_DATASET = "openai/openai_humaneval"
MODELSCOPE_MBPP_DATASET = "OmniData/MBPP"
MODELSCOPE_HUMANEVAL_DATASET = "openai-mirror/openai_humaneval"
DATASET_SOURCES = {"auto", "modelscope", "huggingface", "offline"}


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


def build_mbpp_tasks(
    output_dir: Path,
    limit: int = 20,
    split: str = "test",
    source_path: Path | None = None,
    dataset_source: str = "auto",
    modelscope_dataset: str = MODELSCOPE_MBPP_DATASET,
) -> BuildSummary:
    rows = (
        list(read_jsonl(source_path))
        if source_path
        else load_dataset_rows(
            hf_dataset_name=HF_MBPP_DATASET,
            split=split,
            fallback=OFFLINE_MBPP_ROWS,
            dataset_source=dataset_source,
            modelscope_dataset_name=modelscope_dataset,
        )
    )
    repo_root = output_dir / "repos"
    manifest_path = output_dir / "tasks.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root.mkdir(parents=True, exist_ok=True)
    count = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in rows[:limit]:
            task_id = safe_id(f"mbpp_{row.get('task_id', count)}")
            code = str(row.get("code", "")).strip()
            tests = row.get("test_list") or []
            if not code or not isinstance(tests, list):
                continue
            function_name = extract_function_name(code) or "solution"
            skeleton = python_skeleton_from_solution(code)
            repo_path = repo_root / task_id
            write_unit_repo(repo_path, function_name, skeleton, [str(test) for test in tests])
            manifest.write(
                json.dumps(
                    {
                        "id": task_id,
                        "source": "mbpp",
                        "repo": f"repos/{task_id}",
                        "task": f"Implement `{function_name}` in solution.py so it satisfies the task: {row.get('text', '')}",
                        "target_file": "solution.py",
                        "target_symbol": function_name,
                        "test_command": "python3 -m unittest discover -s tests",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    return BuildSummary(count=count, output_path=manifest_path, extra_path=repo_root)


def build_humaneval_tasks(
    output_dir: Path,
    limit: int = 20,
    split: str = "test",
    source_path: Path | None = None,
    dataset_source: str = "auto",
    modelscope_dataset: str = MODELSCOPE_HUMANEVAL_DATASET,
) -> BuildSummary:
    rows = (
        list(read_jsonl(source_path))
        if source_path
        else load_dataset_rows(
            hf_dataset_name=HF_HUMANEVAL_DATASET,
            split=split,
            fallback=OFFLINE_HUMANEVAL_ROWS,
            dataset_source=dataset_source,
            modelscope_dataset_name=modelscope_dataset,
        )
    )
    repo_root = output_dir / "repos"
    manifest_path = output_dir / "tasks.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root.mkdir(parents=True, exist_ok=True)
    count = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in rows[:limit]:
            entry_point = str(row.get("entry_point", "")).strip()
            prompt = str(row.get("prompt", ""))
            test = str(row.get("test", ""))
            if not entry_point or not prompt or not test:
                continue
            task_id = safe_id(str(row.get("task_id", f"humaneval_{count}")))
            repo_path = repo_root / task_id
            write_humaneval_repo(repo_path, prompt, entry_point, test)
            manifest.write(
                json.dumps(
                    {
                        "id": task_id,
                        "source": "humaneval",
                        "repo": f"repos/{task_id}",
                        "task": f"Implement `{entry_point}` in solution.py so it passes the HumanEval-style tests.",
                        "target_file": "solution.py",
                        "target_symbol": entry_point,
                        "test_command": "python3 -m unittest discover -s tests",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    return BuildSummary(count=count, output_path=manifest_path, extra_path=repo_root)


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


def write_humaneval_repo(repo_path: Path, prompt: str, entry_point: str, test: str) -> None:
    repo_path.mkdir(parents=True, exist_ok=True)
    solution = prompt.rstrip() + "\n    pass\n"
    (repo_path / "solution.py").write_text(solution, encoding="utf-8")
    test_content = (
        "import unittest\n\n"
        f"from solution import {entry_point}\n\n"
        f"{test.rstrip()}\n\n"
        "class GeneratedTests(unittest.TestCase):\n"
        "    def test_humaneval(self) -> None:\n"
        f"        check({entry_point})\n"
    )
    (repo_path / "tests").mkdir(exist_ok=True)
    (repo_path / "tests" / "test_solution.py").write_text(test_content, encoding="utf-8")
    (repo_path / "AGENT.md").write_text("Implement the target function. Run tests after editing.\n", encoding="utf-8")


def load_dataset_rows(
    hf_dataset_name: str,
    split: str,
    fallback: list[dict[str, Any]],
    dataset_source: str = "auto",
    modelscope_dataset_name: str | None = None,
) -> list[dict[str, Any]]:
    if dataset_source not in DATASET_SOURCES:
        raise ValueError(f"unknown dataset source: {dataset_source}")
    if dataset_source == "offline":
        return fallback

    if dataset_source in {"auto", "modelscope"} and modelscope_dataset_name:
        rows = load_modelscope_rows(modelscope_dataset_name, split)
        if rows or dataset_source == "modelscope":
            return rows or fallback

    if dataset_source in {"auto", "huggingface"}:
        rows = load_huggingface_rows(hf_dataset_name, split)
        if rows or dataset_source == "huggingface":
            return rows or fallback

    return fallback


def load_modelscope_rows(dataset_name: str, split: str) -> list[dict[str, Any]]:
    if dataset_name == MODELSCOPE_MBPP_DATASET:
        rows = load_modelscope_jsonl_file(dataset_name, "raw/mbpp.jsonl")
        if rows:
            return rows

    try:
        from modelscope.msdatasets import MsDataset  # type: ignore
    except Exception:
        return []

    token = os.getenv("MODELSCOPE_SDK_TOKEN") or os.getenv("MODELSCOPE_API_TOKEN")
    if token and not os.getenv("MODELSCOPE_SDK_TOKEN"):
        os.environ["MODELSCOPE_SDK_TOKEN"] = token

    for candidate_split in unique_values([split, "train", "test", "validation", "dev"]):
        try:
            dataset = MsDataset.load(dataset_name, split=candidate_split)
            rows = rows_from_loaded_dataset(dataset, candidate_split)
            if rows:
                return rows
        except Exception:
            continue

    try:
        return rows_from_loaded_dataset(MsDataset.load(dataset_name), split)
    except Exception:
        return []


def load_modelscope_jsonl_file(dataset_name: str, file_path: str) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"Source": "SDK", "Revision": "master", "FilePath": file_path, "View": "False"})
    url = f"https://www.modelscope.cn/api/v1/datasets/{dataset_name}/repo?{query}"
    request = urllib.request.Request(url)
    token = os.getenv("MODELSCOPE_SDK_TOKEN") or os.getenv("MODELSCOPE_API_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode("utf-8")
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return []
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_huggingface_rows(dataset_name: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore

        return rows_from_loaded_dataset(load_dataset(dataset_name, split=split), split)
    except Exception:
        return []


def rows_from_loaded_dataset(dataset: Any, split: str) -> list[dict[str, Any]]:
    if isinstance(dataset, dict):
        dataset = dataset.get(split) or dataset.get("train") or next(iter(dataset.values()), [])
    rows: list[dict[str, Any]] = []
    for row in dataset:
        if isinstance(row, dict):
            rows.append(dict(row))
    return rows


def unique_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def extract_function_name(code: str) -> str | None:
    match = re.search(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code, flags=re.MULTILINE)
    return match.group(1) if match else None


def python_skeleton_from_solution(code: str) -> str:
    match = re.search(r"^def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^\n]*\)\s*(?:->\s*[^:]+)?:", code, flags=re.MULTILINE)
    if not match:
        return "def solution(*args, **kwargs):\n    pass\n"
    return match.group(0) + "\n    pass\n"


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
