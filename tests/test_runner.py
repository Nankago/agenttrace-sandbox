import unittest
import json
from pathlib import Path

from agenttrace_sandbox.config import AgentConfig
from agenttrace_sandbox.data_builders import (
    build_benchmark_tasks,
    build_humaneval_tasks,
    build_mbpp_tasks,
    build_pr_wiki,
    build_unit_completion_tasks,
    bug_fix_quality,
    discover_test_function_refs,
    is_bug_fix_record,
    linked_issue_number,
    load_dataset_rows,
    rows_from_loaded_dataset,
)
from agenttrace_sandbox.llm import MockCodingModel
from agenttrace_sandbox.manifest import run_manifest
from agenttrace_sandbox.runner import run_task
from agenttrace_sandbox.sandbox import build_docker_command
from agenttrace_sandbox.sft_export import export_sft
from agenttrace_sandbox.stats import compute_manifest_stats, compute_run_stats


class BadJsonModel:
    def complete(self, system: str, user: str) -> str:
        if "Create a concise plan" in user:
            return "try something"
        return "not json"


class RunnerTests(unittest.TestCase):
    def test_mock_agent_collects_success_trace(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            tmp_path = Path(temp)
            repo = Path("examples/buggy_calculator").resolve()
            config = AgentConfig(runs_dir=tmp_path / "runs", max_steps=8)

            result = run_task(
                repo,
                "Fix the subtract function bug.",
                "python3 -m unittest discover -s tests",
                config,
                MockCodingModel(),
            )

            self.assertEqual(result.outcome, "success")
            self.assertTrue(result.trace_path.exists())
            fixed = result.workspace / "calculator.py"
            self.assertIn("return a - b", fixed.read_text(encoding="utf-8"))

            sft_path = tmp_path / "sft.jsonl"
            count = export_sft(tmp_path / "runs", sft_path)
            self.assertGreaterEqual(count, 3)
            self.assertIn('"tool": "replace_in_file"', sft_path.read_text(encoding="utf-8"))

            alpaca_path = tmp_path / "sft_alpaca.json"
            alpaca_count = export_sft(tmp_path / "runs", alpaca_path, output_format="alpaca")
            self.assertEqual(alpaca_count, count)
            alpaca_rows = json.loads(alpaca_path.read_text(encoding="utf-8"))
            self.assertIn("system", alpaca_rows[0])

            stats = compute_run_stats(tmp_path / "runs")
            self.assertEqual(stats.total_runs, 1)
            self.assertEqual(stats.outcomes.get("success"), 1)
            self.assertEqual(stats.backends.get("local"), 1)

    def test_invalid_json_is_classified(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            tmp_path = Path(temp)
            repo = Path("examples/buggy_calculator").resolve()
            config = AgentConfig(runs_dir=tmp_path / "runs", max_steps=3, json_retries=1)

            result = run_task(
                repo,
                "Fix the subtract function bug.",
                "python3 -m unittest discover -s tests",
                config,
                BadJsonModel(),
            )

            self.assertEqual(result.outcome, "invalid_json")
            self.assertIn("Invalid model JSON", result.final_summary)

    def test_manifest_runner_collects_results(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            tmp_path = Path(temp)
            manifest = tmp_path / "tasks.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "sample",
                        "repo": str(Path("examples/buggy_calculator").resolve()),
                        "task": "Fix the subtract function bug.",
                        "test_command": "python3 -m unittest discover -s tests",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = AgentConfig(runs_dir=tmp_path / "runs", max_steps=8)
            output = tmp_path / "results.jsonl"

            summary = run_manifest(manifest, config, MockCodingModel(), output)

            self.assertEqual(summary.total, 1)
            self.assertEqual(summary.ran, 1)
            self.assertEqual(summary.outcomes.get("success"), 1)
            self.assertIn('"outcome": "success"', output.read_text(encoding="utf-8"))

    def test_docker_command_is_restricted(self) -> None:
        command = build_docker_command(
            workspace=Path("/tmp/workspace"),
            command="python3 -m unittest discover -s tests",
            image="python:3.11-slim",
            network="none",
            memory="512m",
            cpus="0.5",
        )

        self.assertEqual(command[:3], ["docker", "run", "--rm"])
        self.assertIn("--network", command)
        self.assertIn("none", command)
        self.assertIn("--memory", command)
        self.assertIn("512m", command)
        self.assertIn("--cpus", command)
        self.assertIn("0.5", command)
        self.assertIn("/tmp/workspace:/workspace", command)
        self.assertEqual(command[-3:], ["sh", "-lc", "python3 -m unittest discover -s tests"])

    def test_build_benchmark_tasks(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = build_benchmark_tasks(root / "benchmarks", limit=2)
            manifest = summary.output_path
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary.count, 2)
            self.assertEqual(len(rows), 2)
            repo = manifest.parent / rows[0]["repo"]
            self.assertTrue((repo / "solution.py").exists())
            self.assertTrue((repo / "tests" / "test_solution.py").exists())

            dry_run_summary = run_manifest(manifest, AgentConfig(runs_dir=root / "runs"), MockCodingModel(), root / "results.jsonl", dry_run=True)
            self.assertEqual(dry_run_summary.total, 2)
            self.assertEqual(dry_run_summary.skipped, 2)

    def test_build_mbpp_tasks(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = build_mbpp_tasks(root / "mbpp", limit=1, dataset_source="offline")
            manifest = summary.output_path
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary.count, 1)
            self.assertEqual(rows[0]["source"], "mbpp")
            repo = manifest.parent / rows[0]["repo"]
            self.assertIn("pass", (repo / "solution.py").read_text(encoding="utf-8"))
            self.assertIn("assert", (repo / "tests" / "test_solution.py").read_text(encoding="utf-8"))

            dry_run_summary = run_manifest(manifest, AgentConfig(runs_dir=root / "runs"), MockCodingModel(), root / "results.jsonl", dry_run=True)
            self.assertEqual(dry_run_summary.total, 1)
            self.assertEqual(dry_run_summary.skipped, 1)

    def test_build_humaneval_tasks(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = build_humaneval_tasks(root / "humaneval", limit=1, dataset_source="offline")
            manifest = summary.output_path
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary.count, 1)
            self.assertEqual(rows[0]["source"], "humaneval")
            repo = manifest.parent / rows[0]["repo"]
            self.assertIn("pass", (repo / "solution.py").read_text(encoding="utf-8"))
            self.assertIn("check(", (repo / "tests" / "test_solution.py").read_text(encoding="utf-8"))

            dry_run_summary = run_manifest(manifest, AgentConfig(runs_dir=root / "runs"), MockCodingModel(), root / "results.jsonl", dry_run=True)
            self.assertEqual(dry_run_summary.total, 1)
            self.assertEqual(dry_run_summary.skipped, 1)

    def test_build_unit_completion_tasks(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            (repo / "tests").mkdir(parents=True)
            (repo / "math_utils.py").write_text(
                "def add(a, b):\n"
                "    return a + b\n\n"
                "def hidden(a, b):\n"
                "    return a * b\n",
                encoding="utf-8",
            )
            (repo / "tests" / "test_math_utils.py").write_text(
                "import unittest\n\n"
                "from math_utils import add\n\n"
                "class MathTests(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(add(2, 3), 5)\n",
                encoding="utf-8",
            )

            summary = build_unit_completion_tasks(repo, root / "unit_tasks", limit=5)
            rows = [json.loads(line) for line in summary.output_path.read_text(encoding="utf-8").splitlines()]
            task_repo = summary.output_path.parent / rows[0]["repo"]

            self.assertEqual(summary.count, 1)
            self.assertEqual(rows[0]["source"], "unit_completion")
            self.assertIn("pass", (task_repo / "math_utils.py").read_text(encoding="utf-8"))

            dry_run_summary = run_manifest(summary.output_path, AgentConfig(runs_dir=root / "runs"), MockCodingModel(), root / "results.jsonl", dry_run=True)
            self.assertEqual(dry_run_summary.total, 1)
            self.assertEqual(dry_run_summary.skipped, 1)

    def test_discover_test_function_refs_import_patterns(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_refs.py").write_text(
                "from math_utils import add\n"
                "from math_utils import subtract as minus\n"
                "from pkg import math_utils\n"
                "import string_utils\n"
                "import number_utils as nums\n\n"
                "import pkg.strings as strings\n\n"
                "def test_calls():\n"
                "    add(1, 2)\n"
                "    minus(3, 1)\n"
                "    math_utils.multiply(2, 4)\n"
                "    string_utils.slugify('Hello')\n"
                "    nums.double(2)\n"
                "    strings.clean('Hello')\n",
                encoding="utf-8",
            )

            refs = {(ref["module"], ref["function"], ref["access_path"]) for ref in discover_test_function_refs(tests)}

            self.assertIn(("math_utils", "add", "add"), refs)
            self.assertIn(("math_utils", "subtract", "minus"), refs)
            self.assertIn(("pkg.math_utils", "multiply", "math_utils.multiply"), refs)
            self.assertIn(("string_utils", "slugify", "string_utils.slugify"), refs)
            self.assertIn(("number_utils", "double", "nums.double"), refs)
            self.assertIn(("pkg.strings", "clean", "strings.clean"), refs)

    def test_build_unit_completion_import_call_patterns_and_docstring(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            (repo / "tests").mkdir(parents=True)
            (repo / "math_utils.py").write_text(
                "def add(a, b):\n"
                "    \"\"\"Add two numbers.\"\"\"\n"
                "    return a + b\n\n"
                "def subtract(a, b):\n"
                "    return a - b\n\n"
                "def multiply(a, b):\n"
                "    return a * b\n\n"
                "def divide(a, b):\n"
                "    return a / b\n\n",
                encoding="utf-8",
            )
            (repo / "tests" / "test_math_utils.py").write_text(
                "from math_utils import add\n"
                "from math_utils import subtract as minus\n"
                "import math_utils\n"
                "import math_utils as mu\n\n"
                "def test_math():\n"
                "    add(1, 2)\n"
                "    minus(3, 1)\n"
                "    math_utils.multiply(2, 3)\n"
                "    mu.divide(4, 2)\n",
                encoding="utf-8",
            )

            summary = build_unit_completion_tasks(repo, root / "unit_tasks", limit=10)
            rows = [json.loads(line) for line in summary.output_path.read_text(encoding="utf-8").splitlines()]
            symbols = {row["target_symbol"] for row in rows}
            add_row = next(row for row in rows if row["target_symbol"] == "add")
            add_repo = summary.output_path.parent / add_row["repo"]
            add_source = (add_repo / "math_utils.py").read_text(encoding="utf-8")

            self.assertEqual(summary.count, 4)
            self.assertEqual(symbols, {"add", "subtract", "multiply", "divide"})
            self.assertEqual(add_row["original_module"], "math_utils")
            self.assertIn("test_files", add_row)
            self.assertIn('"""Add two numbers."""', add_source)
            self.assertIn("pass", add_source)

    def test_build_unit_completion_from_package_import_and_src_layout(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            (repo / "src" / "pkg").mkdir(parents=True)
            (repo / "tests").mkdir()
            (repo / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (repo / "src" / "pkg" / "math_utils.py").write_text(
                "def add(a, b):\n"
                "    return a + b\n",
                encoding="utf-8",
            )
            (repo / "tests" / "test_pkg.py").write_text(
                "from pkg import math_utils\n\n"
                "def test_add():\n"
                "    math_utils.add(1, 2)\n",
                encoding="utf-8",
            )

            summary = build_unit_completion_tasks(repo, root / "unit_tasks", limit=5)
            rows = [json.loads(line) for line in summary.output_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary.count, 1)
            self.assertEqual(rows[0]["target_file"], "src/pkg/math_utils.py")
            self.assertEqual(rows[0]["original_import"], "from pkg import math_utils")

    def test_build_unit_completion_private_and_max_per_file(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            (repo / "tests").mkdir(parents=True)
            (repo / "math_utils.py").write_text(
                "def _hidden():\n"
                "    return 1\n\n"
                "def first():\n"
                "    return 1\n\n"
                "def second():\n"
                "    return 2\n\n",
                encoding="utf-8",
            )
            (repo / "tests" / "test_math_utils.py").write_text(
                "from math_utils import _hidden, first, second\n\n"
                "def test_values():\n"
                "    _hidden()\n"
                "    first()\n"
                "    second()\n",
                encoding="utf-8",
            )

            summary = build_unit_completion_tasks(repo, root / "unit_tasks", limit=10, max_per_file=1)
            rows = [json.loads(line) for line in summary.output_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary.count, 1)
            self.assertEqual(rows[0]["target_symbol"], "first")
            self.assertNotEqual(rows[0]["target_symbol"], "_hidden")

    def test_build_unit_completion_include_class_method(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            (repo / "tests").mkdir(parents=True)
            (repo / "calculator.py").write_text(
                "class Calculator:\n"
                "    @staticmethod\n"
                "    def add(a, b):\n"
                "        \"\"\"Add two values.\"\"\"\n"
                "        return a + b\n",
                encoding="utf-8",
            )
            (repo / "tests" / "test_calculator.py").write_text(
                "from calculator import Calculator\n\n"
                "def test_add():\n"
                "    Calculator.add(1, 2)\n",
                encoding="utf-8",
            )

            skipped = build_unit_completion_tasks(repo, root / "skipped", include_methods=False)
            summary = build_unit_completion_tasks(repo, root / "unit_tasks", include_methods=True)
            rows = [json.loads(line) for line in summary.output_path.read_text(encoding="utf-8").splitlines()]
            task_repo = summary.output_path.parent / rows[0]["repo"]
            source = (task_repo / "calculator.py").read_text(encoding="utf-8")

            self.assertEqual(skipped.count, 0)
            self.assertEqual(summary.count, 1)
            self.assertEqual(rows[0]["target_class"], "Calculator")
            self.assertIn('"""Add two values."""', source)
            self.assertIn("pass", source)

    def test_dataset_loader_offline_and_split_dict(self) -> None:
        rows = load_dataset_rows("unused", "test", [{"id": 1}], dataset_source="offline")
        self.assertEqual(rows, [{"id": 1}])

        loaded = rows_from_loaded_dataset({"train": [{"id": "train"}], "test": [{"id": "test"}]}, "test")
        self.assertEqual(loaded, [{"id": "test"}])

    def test_build_pr_wiki(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "prs.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "sample_pr",
                        "repo": "owner/repo",
                        "issue_title": "subtract is wrong",
                        "issue_body": "subtract returns the sum.",
                        "pr_title": "Fix subtract",
                        "bug_fix_score": 3,
                        "bug_fix_reasons": ["positive text keywords: Fix", "touches source files"],
                        "diff": "diff --git a/calculator.py b/calculator.py\n--- a/calculator.py\n+++ b/calculator.py\n@@\n-return a + b\n+return a - b",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "wiki.jsonl"
            summary = build_pr_wiki(source, output)
            row = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(summary.count, 1)
            self.assertEqual(row["files"], ["calculator.py"])
            self.assertIn("bug_summary", row["wiki"])
            self.assertEqual(row["metadata"]["bug_fix_score"], 3)
            self.assertIn("bug_fix_reasons", row["wiki"]["source_context"])

    def test_linked_issue_number(self) -> None:
        self.assertEqual(linked_issue_number("Fixed #37102"), 37102)
        self.assertEqual(linked_issue_number("Update parser", "Fixes #123 and updates docs"), 123)
        self.assertEqual(linked_issue_number("Follow up for #456"), 456)
        self.assertEqual(linked_issue_number("No linked issue"), None)

    def test_bug_fix_quality_filters_docs_only(self) -> None:
        record = {
            "pr_title": "Fix typo in README",
            "files": ["README.md"],
            "diff": "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@\n-teh\n+the",
        }

        quality = bug_fix_quality(record)

        self.assertFalse(quality["is_bug_fix"])
        self.assertTrue(quality["docs_only"])
        self.assertFalse(is_bug_fix_record(record))

    def test_bug_fix_quality_accepts_source_and_tests(self) -> None:
        record = {
            "pr_title": "Fix ValueError when parsing empty input",
            "issue_title": "Parser crashes on empty input",
            "issue_body": "A ValueError traceback is raised.",
            "files": ["src/parser.py", "tests/test_parser.py"],
            "issue_number": 123,
            "diff": (
                "diff --git a/src/parser.py b/src/parser.py\n"
                "--- a/src/parser.py\n"
                "+++ b/src/parser.py\n"
                "@@\n"
                "-raise ValueError\n"
                "+return None\n"
                "diff --git a/tests/test_parser.py b/tests/test_parser.py\n"
                "--- a/tests/test_parser.py\n"
                "+++ b/tests/test_parser.py\n"
                "@@\n"
                "+def test_empty_input(): pass"
            ),
        }

        quality = bug_fix_quality(record)

        self.assertTrue(quality["is_bug_fix"])
        self.assertEqual(quality["source_files"], ["src/parser.py"])
        self.assertEqual(quality["test_files"], ["tests/test_parser.py"])
        self.assertFalse(quality["tests_only"])

    def test_bug_fix_quality_filters_tests_only(self) -> None:
        record = {
            "pr_title": "Fix flaky parser test",
            "files": ["tests/test_parser.py"],
            "diff": "diff --git a/tests/test_parser.py b/tests/test_parser.py\n--- a/tests/test_parser.py\n+++ b/tests/test_parser.py\n@@\n+def test_parser_error(): pass",
        }

        quality = bug_fix_quality(record)

        self.assertFalse(quality["is_bug_fix"])
        self.assertTrue(quality["tests_only"])

    def test_strict_sft_export_filters_noisy_traces(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good = root / "runs" / "good"
            noisy = root / "runs" / "noisy"
            good.mkdir(parents=True)
            noisy.mkdir(parents=True)
            write_trace(
                good / "trace.jsonl",
                outcome="success",
                format_violation=False,
                diff="--- a/solution.py\n+++ b/solution.py\n@@\n-pass\n+return 1",
            )
            write_trace(
                noisy / "trace.jsonl",
                outcome="success",
                format_violation=True,
                diff="--- a/solution.py\n+++ b/solution.py\n@@\n-pass\n+return 1",
            )

            output = root / "strict.jsonl"
            count = export_sft(root / "runs", output, strict=True)

            self.assertEqual(count, 2)
            self.assertIn('"trace":', output.read_text(encoding="utf-8"))

            clean_output = root / "clean.jsonl"
            clean_count = export_sft(root / "runs", clean_output, clean_steps=True)

            self.assertEqual(clean_count, 3)

    def test_manifest_stats_group_by_source(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "results.jsonl"
            rows = [
                {"source": {"source": "mbpp"}, "result": {"skipped": False, "outcome": "success"}},
                {"source": {"source": "mbpp"}, "result": {"skipped": False, "outcome": "test_failed"}},
                {"source": {"source": "humaneval"}, "result": {"skipped": True}},
            ]
            results.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            stats = compute_manifest_stats(results)

            self.assertEqual(stats.total, 3)
            self.assertEqual(stats.ran, 2)
            self.assertEqual(stats.skipped, 1)
            self.assertEqual(stats.by_source["mbpp"]["pass_rate"], 0.5)


def write_trace(path: Path, outcome: str, format_violation: bool, diff: str) -> None:
    events = [
        {"event": "run_started", "payload": {"task": "Implement solution."}},
        {"event": "plan", "payload": {"plan": "Inspect, edit, test."}},
        {
            "event": "tool_call",
            "payload": {
                "step": 1,
                "tool": "read_file",
                "arguments": {"path": "solution.py"},
                "reason": "inspect",
                "format_violation": format_violation,
                "retries_used": 0,
                "parse_error": "",
                "result": {"ok": True, "error_type": "", "blocked": False},
            },
        },
        {
            "event": "tool_call",
            "payload": {
                "step": 2,
                "tool": "run_tests",
                "arguments": {"command": "python3 -m unittest discover -s tests"},
                "reason": "validate",
                "format_violation": False,
                "retries_used": 0,
                "parse_error": "",
                "result": {"ok": True, "error_type": "", "blocked": False},
            },
        },
        {"event": "run_finished", "payload": {"outcome": outcome, "diff": diff}},
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
