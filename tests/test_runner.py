import unittest
import json
from pathlib import Path

from agenttrace_sandbox.config import AgentConfig
from agenttrace_sandbox.llm import MockCodingModel
from agenttrace_sandbox.manifest import run_manifest
from agenttrace_sandbox.runner import run_task
from agenttrace_sandbox.sft_export import export_sft


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
