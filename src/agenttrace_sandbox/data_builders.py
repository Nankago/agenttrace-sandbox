from __future__ import annotations

import ast
import json
import os
import re
import shutil
import shlex
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenttrace_sandbox.tracing import read_jsonl
from agenttrace_sandbox.sandbox import pythonpath_env


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
GITHUB_API = "https://api.github.com"
BUG_FIX_POSITIVE_KEYWORDS = [
    "fix",
    "fixed",
    "fixes",
    "bug",
    "regression",
    "error",
    "exception",
    "crash",
    "failing",
    "failure",
    "fail",
    "broken",
    "incorrect",
    "wrong",
    "issue",
    "defect",
    "flaky",
    "traceback",
    "TypeError",
    "ValueError",
    "AssertionError",
]
BUG_FIX_NEGATIVE_KEYWORDS = [
    "docs",
    "documentation",
    "typo",
    "spelling",
    "refactor",
    "cleanup",
    "style",
    "formatting",
    "dependency bump",
    "release",
    "changelog",
    "ci",
    "test-only",
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


def build_unit_completion_tasks(
    source_repo: Path,
    output_dir: Path,
    limit: int = 20,
    tests_dir: str = "tests",
    test_command: str = "python3 -m unittest discover -s tests",
    min_confidence: float = 0.5,
    include_methods: bool = False,
    exclude_private: bool = True,
    max_per_file: int = 5,
    check_baseline: bool = True,
    baseline_timeout: int = 30,
) -> BuildSummary:
    targets = discover_unit_completion_targets(
        source_repo,
        tests_dir=tests_dir,
        min_confidence=min_confidence,
        include_methods=include_methods,
        exclude_private=exclude_private,
        max_per_file=max_per_file,
    )[:limit]
    repo_root = output_dir / "repos"
    manifest_path = output_dir / "tasks.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root.mkdir(parents=True, exist_ok=True)
    count = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for target in targets:
            task_id = safe_id(f"unit_{target['module']}_{target.get('target_class', '')}_{target['function']}")
            repo_path = repo_root / task_id
            if repo_path.exists():
                shutil.rmtree(repo_path)
            shutil.copytree(source_repo, repo_path, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "runs", "data"))
            target_path = repo_path / target["path"]
            if not validate_unit_completion_repo(repo_path, target_path, tests_dir):
                continue
            relevant_test_command = unit_completion_test_command(target, test_command)
            baseline = run_baseline_check(repo_path, relevant_test_command, baseline_timeout) if check_baseline else {
                "checked": False,
                "ok": True,
                "command": relevant_test_command,
                "output": "",
            }
            if check_baseline and not baseline["ok"]:
                continue
            if not blank_function_body(target_path, target["function"], target.get("target_class")):
                continue
            if not validate_unit_completion_repo(repo_path, target_path, tests_dir):
                continue
            row = {
                "id": task_id,
                "source": "unit_completion",
                "repo": f"repos/{task_id}",
                "task": f"Complete `{target['function']}` in `{target['path']}` so the existing unit tests pass.",
                "target_file": target["path"],
                "target_symbol": target["function"],
                "test_files": target["test_files"],
                "test_selectors": target.get("test_selectors", []),
                "confidence": target["confidence"],
                "original_module": target["module"],
                "original_import": target["original_import"],
                "test_command": relevant_test_command,
                "full_test_command": test_command,
                "baseline_checked": baseline["checked"],
                "baseline_ok": baseline["ok"],
                "baseline_command": baseline["command"],
            }
            if baseline["output"]:
                row["baseline_output_excerpt"] = compact_text(baseline["output"])[:500]
            if target.get("target_class"):
                row["target_class"] = target["target_class"]
            manifest.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )
            count += 1

    return BuildSummary(count=count, output_path=manifest_path, extra_path=repo_root)


def fetch_github_prs(
    repo_full_name: str,
    output_path: Path,
    limit: int = 20,
    state: str = "closed",
    bug_fix_only: bool = False,
    min_bug_score: float = 2,
    include_docs_only: bool = False,
    include_tests_only: bool = False,
) -> BuildSummary:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pulls = github_api_json(f"/repos/{repo_full_name}/pulls", {"state": state, "per_page": str(min(limit, 100)), "sort": "updated", "direction": "desc"})
    if not isinstance(pulls, list):
        pulls = []

    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for pr in pulls[:limit]:
            if not isinstance(pr, dict):
                continue
            number = int(pr.get("number") or 0)
            if not number:
                continue
            body = str(pr.get("body") or "")
            title = str(pr.get("title") or "")
            issue_number = linked_issue_number(title, body)
            issue = github_api_json(f"/repos/{repo_full_name}/issues/{issue_number}") if issue_number else {}
            diff = github_api_text(f"/repos/{repo_full_name}/pulls/{number}", accept="application/vnd.github.v3.diff")
            record = {
                "id": f"{repo_full_name}#{number}",
                "repo": repo_full_name,
                "pr_number": number,
                "pr_title": title,
                "pr_body": body,
                "pr_url": pr.get("html_url", ""),
                "issue_number": issue_number,
                "issue_title": issue.get("title", "") if isinstance(issue, dict) else "",
                "issue_body": issue.get("body", "") if isinstance(issue, dict) else "",
                "diff": diff,
                "files": files_from_diff(diff),
            }
            record.update(bug_fix_quality(record))
            if bug_fix_only:
                allowed_structural_exception = (record["docs_only"] and include_docs_only) or (record["tests_only"] and include_tests_only)
                if record["bug_fix_score"] < min_bug_score or (not record["is_bug_fix"] and not allowed_structural_exception):
                    continue
                if record["docs_only"] and not include_docs_only:
                    continue
                if record["tests_only"] and not include_tests_only:
                    continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return BuildSummary(count=count, output_path=output_path)


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


def discover_unit_completion_targets(
    source_repo: Path,
    tests_dir: str = "tests",
    min_confidence: float = 0.5,
    include_methods: bool = False,
    exclude_private: bool = True,
    max_per_file: int = 5,
) -> list[dict[str, Any]]:
    refs = discover_test_function_refs(source_repo / tests_dir)
    targets: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    per_file: dict[str, int] = {}
    for ref in refs:
        function = str(ref.get("function", ""))
        target_class = str(ref.get("target_class", ""))
        if ref.get("confidence", 0) < min_confidence:
            continue
        if exclude_private and function.startswith("_"):
            continue
        if target_class and not include_methods:
            continue
        module = str(ref.get("module", ""))
        module_path = resolve_module_path(source_repo, module)
        if not module_path and include_methods and ref.get("fallback_class_module") and ref.get("fallback_target_class"):
            module = str(ref["fallback_class_module"])
            target_class = str(ref["fallback_target_class"])
            module_path = resolve_module_path(source_repo, module)
        if not module_path:
            continue
        if target_class and not include_methods:
            continue
        path = str(module_path.relative_to(source_repo))
        if per_file.get(path, 0) >= max_per_file:
            continue
        if target_class:
            if not class_method_exists(module_path, target_class, function):
                continue
        elif function not in top_level_function_names(module_path):
            continue
        key = (path, target_class, function)
        if key in seen:
            continue
        seen.add(key)
        per_file[path] = per_file.get(path, 0) + 1
        target = dict(ref)
        target["module"] = module
        target["path"] = path
        target["test_files"] = relative_test_files(source_repo, target.get("test_files", []))
        target["test_selectors"] = relative_test_selectors(source_repo, target.get("test_selectors", []))
        if target_class:
            target["target_class"] = target_class
        targets.append(target)
    return targets


def discover_test_function_refs(tests_path: Path) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if not tests_path.exists():
        return refs
    for path in sorted(tests_path.rglob("test*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        imports = collect_test_imports(tree)
        for node, class_name, test_name in iter_test_calls(tree):
            ref = ref_from_call(node.func, imports, path, class_name=class_name, test_name=test_name)
            if ref:
                refs.append(ref)
    return unique_function_refs(refs)


def iter_test_calls(tree: ast.AST) -> list[tuple[ast.Call, str, str]]:
    calls: list[tuple[ast.Call, str, str]] = []

    def visit_body(nodes: list[ast.stmt], class_name: str = "") -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                visit_body(node.body, class_name=node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        calls.append((child, class_name, node.name))

    if isinstance(tree, ast.Module):
        visit_body(tree.body)
    return calls


def collect_test_imports(tree: ast.AST) -> dict[str, dict[str, Any]]:
    imports: dict[str, dict[str, Any]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                imports[local] = {
                    "kind": "module",
                    "module": alias.name,
                    "original_import": f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""),
                }
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                imports[local] = {
                    "kind": "from",
                    "module": node.module,
                    "name": alias.name,
                    "original_import": f"from {node.module} import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""),
                }
    return imports


def ref_from_call(
    func: ast.expr,
    imports: dict[str, dict[str, Any]],
    test_file: Path,
    class_name: str = "",
    test_name: str = "",
) -> dict[str, Any] | None:
    test_selector = python_test_selector(test_file, class_name, test_name)
    if isinstance(func, ast.Name):
        imported = imports.get(func.id)
        if imported and imported["kind"] == "from":
            return {
                "module": imported["module"],
                "function": imported["name"],
                "alias": func.id,
                "access_path": func.id,
                "test_file": str(test_file),
                "test_files": [str(test_file)],
                "test_selectors": [test_selector] if test_selector else [],
                "test_class": class_name,
                "test_name": test_name,
                "confidence": 1.0,
                "original_import": imported["original_import"],
            }
    if isinstance(func, ast.Attribute):
        names = dotted_attribute_names(func)
        if len(names) < 2:
            return None
        base = names[0]
        imported = imports.get(base)
        if not imported:
            return None
        if imported["kind"] == "module":
            return {
                "module": imported["module"],
                "function": names[-1],
                "alias": base,
                "access_path": ".".join(names),
                "test_file": str(test_file),
                "test_files": [str(test_file)],
                "test_selectors": [test_selector] if test_selector else [],
                "test_class": class_name,
                "test_name": test_name,
                "confidence": 0.95,
                "original_import": imported["original_import"],
            }
        if imported["kind"] == "from":
            if len(names) == 2:
                return {
                    "module": f"{imported['module']}.{imported['name']}",
                    "function": names[-1],
                    "fallback_class_module": imported["module"],
                    "fallback_target_class": imported["name"],
                    "alias": base,
                    "access_path": ".".join(names),
                    "test_file": str(test_file),
                    "test_files": [str(test_file)],
                    "test_selectors": [test_selector] if test_selector else [],
                    "test_class": class_name,
                    "test_name": test_name,
                    "confidence": 0.9,
                    "original_import": imported["original_import"],
                }
            if len(names) == 3:
                return {
                    "module": imported["module"],
                    "function": names[-1],
                    "target_class": imported["name"],
                    "alias": base,
                    "access_path": ".".join(names),
                    "test_file": str(test_file),
                    "test_files": [str(test_file)],
                    "test_selectors": [test_selector] if test_selector else [],
                    "test_class": class_name,
                    "test_name": test_name,
                    "confidence": 0.75,
                    "original_import": imported["original_import"],
                }
    return None


def dotted_attribute_names(node: ast.expr) -> list[str]:
    names: list[str] = []
    current: ast.expr | None = node
    while isinstance(current, ast.Attribute):
        names.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        names.append(current.id)
    names.reverse()
    return names


def unique_function_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for ref in refs:
        key = (str(ref.get("module", "")), str(ref.get("target_class", "")), str(ref.get("function", "")))
        existing = merged.get(key)
        if not existing:
            merged[key] = dict(ref)
            continue
        existing_files = list(existing.get("test_files", []))
        for test_file in ref.get("test_files", []):
            if test_file not in existing_files:
                existing_files.append(test_file)
        existing["test_files"] = existing_files
        existing_selectors = list(existing.get("test_selectors", []))
        for selector in ref.get("test_selectors", []):
            if selector and selector not in existing_selectors:
                existing_selectors.append(selector)
        existing["test_selectors"] = existing_selectors
        existing["confidence"] = max(float(existing.get("confidence", 0)), float(ref.get("confidence", 0)))
    return list(merged.values())


def imported_functions_from_tests(tests_path: Path) -> dict[str, set[str]]:
    imported: dict[str, set[str]] = {}
    for ref in discover_test_function_refs(tests_path):
        if not ref.get("target_class"):
            imported.setdefault(str(ref["module"]), set()).add(str(ref["function"]))
    return imported


def resolve_module_path(source_repo: Path, module: str) -> Path | None:
    relative = Path(*module.split("."))
    candidates = [
        source_repo / relative.with_suffix(".py"),
        source_repo / relative / "__init__.py",
        source_repo / "src" / relative.with_suffix(".py"),
        source_repo / "src" / relative / "__init__.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def top_level_function_names(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def class_method_exists(path: Path, class_name: str, function_name: str) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return any(isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == function_name for child in node.body)
    return False


def blank_function_body(path: Path, function_name: str, class_name: str | None = None) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        tree = ast.parse("\n".join(lines) + "\n")
    except (OSError, SyntaxError):
        return False
    body_nodes = tree.body
    if class_name:
        body_nodes = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                body_nodes = node.body
                break
    for node in body_nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            body = list(node.body)
            if body and is_docstring_node(body[0]):
                if len(body) == 1:
                    indent = " " * (body[0].col_offset + 4)
                    insert_at = body[0].end_lineno or body[0].lineno
                    lines.insert(insert_at, f"{indent}pass")
                else:
                    replace_body_lines(lines, body[1], body[-1], "pass")
            elif body:
                replace_body_lines(lines, body[0], body[-1], "pass")
            path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
            return True
    return False


def validate_unit_completion_repo(repo_path: Path, target_path: Path, tests_dir: str) -> bool:
    if not target_path.exists() or not (repo_path / tests_dir).exists():
        return False
    try:
        ast.parse(target_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    return True


def unit_completion_test_command(target: dict[str, Any], fallback: str) -> str:
    selectors = [str(selector) for selector in target.get("test_selectors", []) if selector]
    if selectors:
        return "python3 -m unittest " + " ".join(selectors)
    test_files = [str(path) for path in target.get("test_files", []) if path]
    modules = [test_module_from_path(Path(path)) for path in test_files]
    modules = [module for module in modules if module]
    if modules:
        return "python3 -m unittest " + " ".join(modules)
    return fallback


def run_baseline_check(repo_path: Path, command: str, timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            shlex.split(command),
            cwd=repo_path,
            env=pythonpath_env(repo_path),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        return {"checked": True, "ok": completed.returncode == 0, "command": command, "output": output}
    except Exception as exc:  # noqa: BLE001
        return {"checked": True, "ok": False, "command": command, "output": f"{type(exc).__name__}: {exc}"}


def python_test_selector(test_file: Path, class_name: str, test_name: str) -> str:
    if not test_name:
        return ""
    module = test_module_from_path(test_file)
    if not module:
        return ""
    parts = [module]
    if class_name:
        parts.append(class_name)
    parts.append(test_name)
    return ".".join(parts)


def test_module_from_path(path: Path) -> str:
    without_suffix = path.with_suffix("")
    return ".".join(without_suffix.parts)


def relative_test_files(source_repo: Path, test_files: Any) -> list[str]:
    values: list[str] = []
    for test_file in test_files if isinstance(test_files, list) else []:
        path = Path(str(test_file))
        try:
            value = path.resolve().relative_to(source_repo.resolve()).as_posix()
        except ValueError:
            value = path.as_posix()
        if value not in values:
            values.append(value)
    return values


def relative_test_selectors(source_repo: Path, selectors: Any) -> list[str]:
    values: list[str] = []
    repo_parts = source_repo.resolve().parts
    for selector in selectors if isinstance(selectors, list) else []:
        value = str(selector)
        parts = value.split(".")
        for index in range(len(repo_parts), 0, -1):
            prefix = ".".join(repo_parts[-index:])
            if value.startswith(prefix + "."):
                value = value[len(prefix) + 1 :]
                break
        marker = ".tests."
        if marker in value:
            value = "tests." + value.split(marker, 1)[1]
        if value not in values:
            values.append(value)
    return values


def is_docstring_node(node: ast.AST) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def replace_body_lines(lines: list[str], first: ast.AST, last: ast.AST, replacement: str) -> None:
    start = first.lineno - 1
    end = (last.end_lineno or last.lineno)
    indent = " " * first.col_offset
    lines[start:end] = [f"{indent}{replacement}"]


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


def github_api_json(path: str, params: dict[str, str] | None = None) -> Any:
    text = github_api_text(path, params=params, accept="application/vnd.github+json")
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def github_api_text(path: str, params: dict[str, str] | None = None, accept: str = "application/vnd.github+json") -> str:
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    request = urllib.request.Request(f"{GITHUB_API}{path}{query}")
    request.add_header("Accept", accept)
    request.add_header("User-Agent", "agenttrace-sandbox")
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def linked_issue_number(title: str, body: str = "") -> int | None:
    text = f"{title}\n{body}"
    strong = re.search(r"\b(?:fix(?:e[sd])?|clos(?:e[sd]?|ing)|resolv(?:e[sd]?|ing))\s+#(\d+)", text, flags=re.IGNORECASE)
    if strong:
        return int(strong.group(1))
    fallback = re.search(r"#(\d+)", text)
    return int(fallback.group(1)) if fallback else None


def is_bug_fix_record(record: dict[str, Any]) -> bool:
    return bool(bug_fix_quality(record)["is_bug_fix"])


def bug_fix_quality(record: dict[str, Any]) -> dict[str, Any]:
    diff = str(record.get("diff", ""))
    files = list(record.get("files") or files_from_diff(diff))
    source_files = [path for path in files if is_source_file(path)]
    test_files = [path for path in files if is_test_file(path)]
    docs_only = bool(files) and all(is_docs_or_config_file(path) for path in files)
    tests_only = bool(files) and bool(test_files) and not source_files
    text = "\n".join(
        [
            str(record.get("pr_title") or ""),
            str(record.get("pr_body") or ""),
            str(record.get("issue_title") or ""),
            str(record.get("issue_body") or ""),
        ]
    )
    diff_text = "\n".join(line for line in diff.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
    positive_text = keyword_hits(text, BUG_FIX_POSITIVE_KEYWORDS)
    positive_diff = keyword_hits(diff_text, BUG_FIX_POSITIVE_KEYWORDS)
    negative_text = keyword_hits(text, BUG_FIX_NEGATIVE_KEYWORDS)

    score = 0.0
    reasons: list[str] = []
    if positive_text:
        score += min(3, len(positive_text))
        reasons.append(f"positive text keywords: {', '.join(positive_text[:5])}")
    if positive_diff:
        score += 1
        reasons.append(f"positive diff keywords: {', '.join(positive_diff[:5])}")
    if record.get("issue_number"):
        score += 1
        reasons.append("linked issue")
    if source_files:
        score += 1
        reasons.append("touches source files")
    if test_files and source_files:
        score += 0.5
        reasons.append("touches tests with source")
    if negative_text:
        score -= 3
        reasons.append(f"negative keywords: {', '.join(negative_text[:5])}")
    if docs_only:
        score -= 4
        reasons.append("docs or CI/config only")
    if tests_only:
        score -= 3
        reasons.append("tests only")

    is_bug_fix = score >= 2 and not docs_only and not tests_only
    return {
        "is_bug_fix": is_bug_fix,
        "bug_fix_score": score,
        "bug_fix_reasons": reasons,
        "source_files": source_files,
        "test_files": test_files,
        "docs_only": docs_only,
        "tests_only": tests_only,
    }


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    hits: list[str] = []
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword.lower()).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, lowered) and keyword not in hits:
            hits.append(keyword)
    return hits


def is_source_file(path: str) -> bool:
    return bool(path) and not is_test_file(path) and not is_docs_or_config_file(path)


def is_test_file(path: str) -> bool:
    lowered = path.lower()
    name = Path(lowered).name
    return (
        lowered.startswith("test/")
        or lowered.startswith("tests/")
        or "/test/" in lowered
        or "/tests/" in lowered
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".test.js")
        or name.endswith(".spec.js")
        or name.endswith(".test.ts")
        or name.endswith(".spec.ts")
    )


def is_docs_or_config_file(path: str) -> bool:
    lowered = path.lower()
    name = Path(lowered).name
    if lowered.startswith(("docs/", "doc/", ".github/", "github/")):
        return True
    if name in {"readme", "readme.md", "changelog", "changelog.md", "changes", "changes.md", "license", "license.md"}:
        return True
    if name.startswith(("readme.", "changelog.", "changes.")):
        return True
    if lowered.endswith((".md", ".rst", ".txt", ".adoc")):
        return True
    return lowered.endswith((".yml", ".yaml")) and ("workflow" in lowered or "ci" in lowered)


def make_pr_wiki(record: dict[str, Any]) -> dict[str, Any]:
    diff = str(record.get("diff", ""))
    files = record.get("files") or files_from_diff(diff)
    issue = compact_text(record.get("issue_body") or record.get("issue") or "")
    pr = compact_text(record.get("pr_body") or record.get("pr") or "")
    title = str(record.get("issue_title") or record.get("pr_title") or record.get("id") or "unknown")
    source_context = {"issue_excerpt": issue[:800], "pr_excerpt": pr[:800]}
    metadata: dict[str, Any] = {}
    if "bug_fix_score" in record:
        metadata["bug_fix_score"] = record.get("bug_fix_score")
        source_context["bug_fix_score"] = record.get("bug_fix_score")
    if "bug_fix_reasons" in record:
        metadata["bug_fix_reasons"] = record.get("bug_fix_reasons")
        source_context["bug_fix_reasons"] = record.get("bug_fix_reasons")
    wiki = {
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
            "source_context": source_context,
        },
    }
    if metadata:
        wiki["metadata"] = metadata
    return wiki


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
