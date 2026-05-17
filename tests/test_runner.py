import unittest
from pathlib import Path

from agenttrace_sandbox.config import AgentConfig
from agenttrace_sandbox.llm import MockCodingModel
from agenttrace_sandbox.runner import run_task
from agenttrace_sandbox.sft_export import export_sft


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
