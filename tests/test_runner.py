import unittest
import json
from pathlib import Path

from agenttrace_sandbox.config import AgentConfig
from agenttrace_sandbox.data_builders import (
    build_benchmark_tasks,
    build_humaneval_tasks,
    build_mbpp_tasks,
    build_pr_wiki,
    build_repair_cards,
    build_unit_completion_tasks,
    bug_fix_quality,
    discover_test_function_refs,
    enrich_repair_cards,
    is_bug_fix_record,
    linked_issue_number,
    load_dataset_rows,
    make_repair_card,
    rows_from_loaded_dataset,
)
from agenttrace_sandbox.llm import MockCodingModel
from agenttrace_sandbox.manifest import run_manifest
from agenttrace_sandbox.runner import run_task
from agenttrace_sandbox.sandbox import build_docker_command
from agenttrace_sandbox.sft_export import clean_repair_text, export_repair_corpus, export_repair_sft, export_sft, make_repair_sft_sample, stats_repair_cards
from agenttrace_sandbox.stats import compute_manifest_stats, compute_run_stats


class BadJsonModel:
    def complete(self, system: str, user: str) -> str:
        if "Create a concise plan" in user:
            return "try something"
        return "not json"


class RepairCardModel:
    def __init__(self, invalid_id: bool = False) -> None:
        self.invalid_id = invalid_id

    def complete(self, system: str, user: str) -> str:
        evidence_id = "EX" if self.invalid_id else "E2"
        return json.dumps(
            {
                "root_cause": {
                    "text": "The parser raises ValueError for empty input instead of returning an empty result.",
                    "evidence_ids": ["E1", evidence_id],
                },
                "failure_condition": {
                    "text": "Calling parse with an empty string triggers the bug.",
                    "evidence_ids": ["E1"],
                },
                "expected_behavior": {
                    "text": "Empty input should return None.",
                    "evidence_ids": ["E3"],
                },
                "repair_rationale": {
                    "text": "Returning None avoids raising ValueError for the covered empty-input case.",
                    "evidence_ids": ["E2"],
                },
                "edge_cases": [
                    {"text": "Empty string input is the regression case.", "evidence_ids": ["E3"]},
                ],
            }
        )


def sample_enriched_repair_card() -> dict:
    return {
        "id": "owner/repo#7",
        "repo": "owner/repo",
        "source_record": {"pr_number": 7, "pr_url": "https://github.com/owner/repo/pull/7", "issue_number": 5},
        "source_files": ["src/parser.py"],
        "test_files": ["tests/test_parser.py"],
        "evidence": [
            {"id": "E1", "type": "issue_text", "text": "Parser crashes on empty input."},
            {"id": "E2", "type": "source_diff", "file": "src/parser.py", "text": "-raise ValueError\n+return None"},
            {"id": "E3", "type": "test_diff", "file": "tests/test_parser.py", "text": "assert parse('') is None"},
        ],
        "repair_card": {
            "symptom": {"text": "Parser crashes on empty input.", "evidence_ids": ["E1"]},
            "localization": {"source_files": ["src/parser.py"], "test_files": ["tests/test_parser.py"], "evidence_ids": ["E2", "E3"]},
            "patch_intent": {"text": "src/parser.py: added lines=1, removed lines=1", "evidence_ids": ["E2"]},
            "test_oracle": {"text": "tests/test_parser.py: added lines=1, removed lines=0", "evidence_ids": ["E3"]},
        },
        "derived_tasks": {
            "repair_instruction": {
                "input": "Use localization and test evidence to write a concise repair instruction.",
                "output": "Update src/parser.py so empty input returns None and the relevant tests pass.",
            }
        },
        "quality": {"overall": 0.9, "has_test_evidence": True, "has_source_patch": True, "bug_fix_score": 4.0},
        "llm_repair_card": {
            "root_cause": {"text": "The parser raises ValueError for empty input.", "evidence_ids": ["E1", "E2"]},
            "failure_condition": {"text": "Calling parse with an empty string triggers the bug.", "evidence_ids": ["E1"]},
            "expected_behavior": {"text": "Empty input should return None.", "evidence_ids": ["E3"]},
            "repair_rationale": {"text": "Returning None avoids raising ValueError for the covered case.", "evidence_ids": ["E2", "E3"]},
            "edge_cases": [{"text": "Empty string input is the regression case.", "evidence_ids": ["E3"]}],
        },
        "llm_quality": {"valid_json": True, "grounding_ok": True, "field_coverage": 1.0},
    }


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
            self.assertEqual(rows[0]["test_command"], "python3 -m unittest tests.test_math_utils.MathTests.test_add")
            self.assertTrue(rows[0]["baseline_ok"])
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
                "import unittest\n\n"
                "from math_utils import add\n"
                "from math_utils import subtract as minus\n"
                "import math_utils\n"
                "import math_utils as mu\n\n"
                "class MathTests(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        add(1, 2)\n"
                "    def test_subtract(self):\n"
                "        minus(3, 1)\n"
                "    def test_multiply(self):\n"
                "        math_utils.multiply(2, 3)\n"
                "    def test_divide(self):\n"
                "        mu.divide(4, 2)\n",
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
            self.assertEqual(add_row["test_selectors"], ["tests.test_math_utils.MathTests.test_add"])
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
                "import unittest\n\n"
                "from pkg import math_utils\n\n"
                "class PkgTests(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        math_utils.add(1, 2)\n",
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
                "import unittest\n\n"
                "from math_utils import _hidden, first, second\n\n"
                "class MathTests(unittest.TestCase):\n"
                "    def test_values(self):\n"
                "        _hidden()\n"
                "        first()\n"
                "        second()\n",
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
                "import unittest\n\n"
                "from calculator import Calculator\n\n"
                "class CalculatorTests(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        Calculator.add(1, 2)\n",
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

    def test_build_unit_completion_baseline_filters_dirty_targets(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = Path("examples/buggy_calculator").resolve()

            summary = build_unit_completion_tasks(repo, root / "unit_tasks", limit=10)
            rows = [json.loads(line) for line in summary.output_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary.count, 1)
            self.assertEqual(rows[0]["target_symbol"], "add")
            self.assertEqual(rows[0]["test_command"], "python3 -m unittest tests.test_calculator.CalculatorTests.test_add")
            self.assertTrue(rows[0]["baseline_checked"])
            self.assertTrue(rows[0]["baseline_ok"])

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

    def test_build_repair_cards(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "prs.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "owner/repo#7",
                        "repo": "owner/repo",
                        "pr_number": 7,
                        "pr_title": "Fix parser ValueError",
                        "pr_body": "Adds a regression test for empty input.",
                        "issue_number": 3,
                        "issue_title": "Parser crashes",
                        "issue_body": "Empty input raises ValueError.",
                        "bug_fix_score": 4,
                        "bug_fix_reasons": ["positive text keywords: Fix, ValueError", "touches tests with source"],
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
                            "+def test_empty_input():\n"
                            "+    assert parse('') is None\n"
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "cards.jsonl"
            summary = build_repair_cards(source, output)
            row = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(summary.count, 1)
            self.assertEqual(row["source_files"], ["src/parser.py"])
            self.assertEqual(row["test_files"], ["tests/test_parser.py"])
            self.assertTrue(row["quality"]["has_test_evidence"])
            self.assertGreater(row["quality"]["overall"], 0.7)
            evidence_types = {item["type"] for item in row["evidence"]}
            self.assertIn("source_diff", evidence_types)
            self.assertIn("test_diff", evidence_types)
            self.assertIn("E", row["repair_card"]["patch_intent"]["evidence_ids"][0])
            self.assertEqual(row["derived_tasks"]["localize_files"]["output"], ["src/parser.py"])

    def test_enrich_repair_cards(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "cards.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "owner/repo#7",
                        "repo": "owner/repo",
                        "evidence": [
                            {"id": "E1", "type": "issue_text", "text": "Parser crashes on empty input."},
                            {"id": "E2", "type": "source_diff", "file": "src/parser.py", "text": "-raise ValueError\n+return None"},
                            {"id": "E3", "type": "test_diff", "file": "tests/test_parser.py", "text": "assert parse('') is None"},
                        ],
                        "repair_card": {
                            "symptom": {"text": "Parser crashes on empty input.", "evidence_ids": ["E1"]},
                            "localization": {"source_files": ["src/parser.py"], "test_files": ["tests/test_parser.py"], "evidence_ids": ["E2", "E3"]},
                        },
                        "quality": {"overall": 0.9},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "enriched.jsonl"
            summary = enrich_repair_cards(source, output, RepairCardModel())
            row = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(summary.count, 1)
            self.assertIn("root_cause", row["llm_repair_card"])
            self.assertTrue(row["llm_quality"]["valid_json"])
            self.assertTrue(row["llm_quality"]["all_evidence_ids_valid"])
            self.assertTrue(row["llm_quality"]["grounding_ok"])
            self.assertGreater(row["llm_quality"]["field_coverage"], 0.9)

    def test_enrich_repair_cards_flags_invalid_evidence(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "cards.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "owner/repo#7",
                        "evidence": [{"id": "E1", "type": "issue_text", "text": "Parser crashes."}],
                        "repair_card": {},
                        "quality": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "enriched.jsonl"
            enrich_repair_cards(source, output, RepairCardModel(invalid_id=True))
            row = json.loads(output.read_text(encoding="utf-8"))

            self.assertFalse(row["llm_quality"]["all_evidence_ids_valid"])
            self.assertIn("EX", row["llm_quality"]["invalid_evidence_ids"])

    def test_export_repair_sft(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "enriched_cards.jsonl"
            source.write_text(json.dumps(sample_enriched_repair_card()) + "\n", encoding="utf-8")
            output = root / "repair_sft.jsonl"

            count = export_repair_sft(source, output, min_quality=0.7, require_grounding=True)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            task_types = {row["metadata"]["task_type"] for row in rows}

            self.assertEqual(count, 5)
            self.assertEqual(task_types, {"localize_files", "explain_bug", "repair_rationale", "test_spec", "repair_instruction"})
            self.assertTrue(all(row["metadata"]["grounding_ok"] for row in rows))
            self.assertIn("E1", rows[0]["input"]["evidence"])

    def test_export_repair_sft_filters_by_quality(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "enriched_cards.jsonl"
            card = sample_enriched_repair_card()
            card["quality"]["overall"] = 0.2
            source.write_text(json.dumps(card) + "\n", encoding="utf-8")
            output = root / "repair_sft.jsonl"

            count = export_repair_sft(source, output, min_quality=0.7)

            self.assertEqual(count, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "")

    def test_stats_repair_cards_basic_quality(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "cards.jsonl"
            card = sample_enriched_repair_card()
            card.pop("llm_repair_card")
            card.pop("llm_quality")
            source.write_text(json.dumps(card) + "\n", encoding="utf-8")

            stats = stats_repair_cards(source)

            self.assertEqual(stats["records"], 1)
            self.assertEqual(stats["quality"]["avg"], 0.9)
            self.assertEqual(stats["has_test_evidence"]["count"], 1)
            self.assertEqual(stats["has_source_patch"]["count"], 1)
            self.assertEqual(stats["avg_bug_fix_score"], 4.0)
            self.assertEqual(stats["derived_tasks"]["repair_instruction"], 1)
            self.assertNotIn("llm_quality", stats)

    def test_stats_repair_cards_enriched_quality(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "enriched_cards.jsonl"
            source.write_text(json.dumps(sample_enriched_repair_card()) + "\n", encoding="utf-8")

            stats = stats_repair_cards(source)

            self.assertEqual(stats["llm_quality"]["valid_json"]["count"], 1)
            self.assertEqual(stats["llm_quality"]["grounding_ok"]["count"], 1)
            self.assertEqual(stats["llm_quality"]["avg_field_coverage"], 1.0)

    def test_export_repair_corpus_writes_linearized_text(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "enriched_cards.jsonl"
            source.write_text(json.dumps(sample_enriched_repair_card()) + "\n", encoding="utf-8")
            output = root / "corpus.jsonl"

            count = export_repair_corpus(source, output, min_quality=0.7, require_grounding=True)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            text = rows[0]["text"]

            self.assertEqual(count, 1)
            self.assertIn("[E1: issue_text", text)
            self.assertIn("Root Cause: The parser raises ValueError for empty input.", text)
            self.assertIn("Test Oracle: tests/test_parser.py: added lines=1, removed lines=0", text)
            self.assertNotIn("-raise ValueError", text)
            self.assertEqual(rows[0]["metadata"]["grounding_ok"], True)

    def test_export_repair_corpus_filters_quality_and_grounding(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "enriched_cards.jsonl"
            low_quality = sample_enriched_repair_card()
            low_quality["quality"]["overall"] = 0.2
            ungrounded = sample_enriched_repair_card()
            ungrounded["id"] = "owner/repo#8"
            ungrounded["llm_quality"]["grounding_ok"] = False
            good = sample_enriched_repair_card()
            good["id"] = "owner/repo#9"
            source.write_text("\n".join(json.dumps(card) for card in [low_quality, ungrounded, good]) + "\n", encoding="utf-8")
            output = root / "corpus.jsonl"

            count = export_repair_corpus(source, output, min_quality=0.7, require_grounding=True)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(count, 1)
            self.assertEqual(rows[0]["id"], "owner/repo#9")

    def test_clean_repair_text_removes_pr_template_noise(self) -> None:
        text = (
            "Fixed #37066 -- Ensure cleanup. "
            "#### Branch description Fix delayed cleanup of async iterables. "
            "#### AI Assistance Disclosure (REQUIRED) <!-- Please select exactly ONE. --> "
            "- [x] No AI tools were used in preparing this PR. "
            "#### Checklist - [x] This PR follows the contribution guidelines. "
            "- [x] This PR targets the `main` branch."
        )

        cleaned = clean_repair_text(text, "light")

        self.assertIn("Branch description Fix delayed cleanup of async iterables", cleaned)
        self.assertNotIn("AI Assistance Disclosure", cleaned)
        self.assertNotIn("Checklist", cleaned)
        self.assertNotIn("contribution guidelines", cleaned)

    def test_clean_repair_text_keep_preserves_template(self) -> None:
        text = "#### AI Assistance Disclosure (REQUIRED) - [x] No AI tools were used."

        self.assertEqual(clean_repair_text(text, "keep"), text)

    def test_clean_repair_text_semantic_keeps_repair_sections(self) -> None:
        text = (
            "Fixed #1 -- Parser empty input. "
            "#### Branch description Return None for empty parser input. "
            "#### Tests Run python -m pytest tests/test_parser.py "
            "#### AI Assistance Disclosure (REQUIRED) - [x] No AI tools were used. "
            "#### Checklist - [x] This PR follows the contribution guidelines."
        )

        cleaned = clean_repair_text(text, "semantic")

        self.assertIn("Return None for empty parser input", cleaned)
        self.assertIn("python -m pytest tests/test_parser.py", cleaned)
        self.assertNotIn("AI Assistance Disclosure", cleaned)
        self.assertNotIn("contribution guidelines", cleaned)

    def test_export_repair_corpus_cleans_boilerplate_by_default(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "enriched_cards.jsonl"
            card = sample_enriched_repair_card()
            card["evidence"][0]["text"] = (
                "Fixed #1 -- Parser empty input. "
                "#### Branch description Return None for empty parser input. "
                "#### AI Assistance Disclosure (REQUIRED) <!-- choose one --> - [x] No AI tools were used. "
                "#### Checklist - [x] This PR follows the contribution guidelines."
            )
            card["repair_card"]["symptom"]["text"] = card["evidence"][0]["text"]
            source.write_text(json.dumps(card) + "\n", encoding="utf-8")
            output = root / "corpus.jsonl"

            export_repair_corpus(source, output, min_quality=0.7, require_grounding=True)
            text = json.loads(output.read_text(encoding="utf-8").splitlines()[0])["text"]

            self.assertIn("Return None for empty parser input", text)
            self.assertNotIn("AI Assistance Disclosure", text)
            self.assertNotIn("contribution guidelines", text)

    def test_export_repair_sft_no_llm_variant_uses_rule_card(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "cards.jsonl"
            card = sample_enriched_repair_card()
            card.pop("llm_repair_card")
            source.write_text(json.dumps(card) + "\n", encoding="utf-8")
            output = root / "repair_sft.jsonl"

            count = export_repair_sft(source, output, tasks=["repair_rationale", "repair_instruction"], variant="no-llm")
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(count, 2)
            self.assertIn("src/parser.py: added lines=1, removed lines=1", rows[0]["output"])
            self.assertEqual(rows[0]["metadata"]["variant"], "no-llm")

    def test_export_repair_sft_no_tests_variant_skips_test_spec_and_test_evidence(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "cards.jsonl"
            source.write_text(json.dumps(sample_enriched_repair_card()) + "\n", encoding="utf-8")
            output = root / "repair_sft.jsonl"

            count = export_repair_sft(source, output, variant="no-tests")
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            task_types = {row["metadata"]["task_type"] for row in rows}

            self.assertEqual(count, 4)
            self.assertNotIn("test_spec", task_types)
            self.assertTrue(all("test_diff" not in row["input"]["evidence"] for row in rows))
            localization = next(row for row in rows if row["metadata"]["task_type"] == "localize_files")
            self.assertEqual(localization["output"]["test_files"], [])

    def test_repair_instruction_is_specific(self) -> None:
        card = sample_enriched_repair_card()

        sample = make_repair_sft_sample(card, "repair_instruction")

        self.assertIsNotNone(sample)
        self.assertIn("src/parser.py", sample["output"])
        self.assertIn("raises ValueError for empty input", sample["output"])
        self.assertNotIn("behavior described by the issue is fixed", sample["output"])

    def test_linked_issue_number(self) -> None:
        self.assertEqual(linked_issue_number("Fixed #37102"), 37102)
        self.assertEqual(linked_issue_number("Update parser", "Fixes #123 and updates docs"), 123)
        self.assertEqual(linked_issue_number("Fix parser", "Closes https://github.com/owner/repo/issues/456"), 456)
        self.assertEqual(linked_issue_number("Follow up for #456"), None)
        self.assertEqual(linked_issue_number("Update parser", "<!-- Example: Fixes #999 -->\nNo linked issue."), None)
        self.assertEqual(linked_issue_number("Update parser", "```text\nFixes #888\n```"), None)
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

    def test_bug_fix_quality_filters_dependency_only(self) -> None:
        record = {
            "pr_title": "Fix dependency resolution",
            "files": ["package-lock.json"],
            "diff": "diff --git a/package-lock.json b/package-lock.json\n--- a/package-lock.json\n+++ b/package-lock.json\n@@\n-1\n+2",
        }

        quality = bug_fix_quality(record)

        self.assertFalse(quality["is_bug_fix"])
        self.assertTrue(quality["dependency_only"])

    def test_repair_card_skips_low_signal_diff_evidence(self) -> None:
        record = {
            "id": "owner/repo#10",
            "repo": "owner/repo",
            "pr_title": "Fix parser empty input",
            "pr_body": "#### Branch description Return None for empty parser input.",
            "issue_number": 10,
            "diff": (
                "diff --git a/src/parser.py b/src/parser.py\n"
                "--- a/src/parser.py\n"
                "+++ b/src/parser.py\n"
                "@@\n"
                "-raise ValueError\n"
                "+return None\n"
                "diff --git a/package-lock.json b/package-lock.json\n"
                "--- a/package-lock.json\n"
                "+++ b/package-lock.json\n"
                "@@\n"
                "-1\n"
                "+2\n"
            ),
        }

        card = make_repair_card(record)
        evidence_files = {item.get("file") for item in card["evidence"]}

        self.assertIn("src/parser.py", evidence_files)
        self.assertNotIn("package-lock.json", evidence_files)

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
