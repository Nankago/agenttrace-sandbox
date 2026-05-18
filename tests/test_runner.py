import unittest
import json
from pathlib import Path

from agenttrace_sandbox.config import AgentConfig
from agenttrace_sandbox.data_builders import (
    build_benchmark_tasks,
    build_humaneval_tasks,
    build_mbpp_tasks,
    build_pr_wiki,
    load_dataset_rows,
    rows_from_loaded_dataset,
)
from agenttrace_sandbox.llm import MockCodingModel
from agenttrace_sandbox.manifest import run_manifest
from agenttrace_sandbox.runner import run_task
from agenttrace_sandbox.sandbox import build_docker_command
from agenttrace_sandbox.sft_export import export_sft
from agenttrace_sandbox.stats import compute_run_stats


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
