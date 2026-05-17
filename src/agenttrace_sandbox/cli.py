from __future__ import annotations

import argparse
from pathlib import Path

from agenttrace_sandbox.config import AgentConfig
from agenttrace_sandbox.llm import MockCodingModel, OpenAICompatibleChat
from agenttrace_sandbox.runner import run_task
from agenttrace_sandbox.sft_export import export_sft


def main() -> None:
    parser = argparse.ArgumentParser(prog="agenttrace", description="Safe trajectory collection for coding-agent post-training.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run one coding-agent task in an isolated workspace.")
    run_parser.add_argument("--repo", required=True, type=Path)
    run_parser.add_argument("--task", required=True)
    run_parser.add_argument("--test-command", default="python3 -m unittest discover -s tests")
    run_parser.add_argument("--mock", action="store_true", help="Use deterministic local model instead of an API.")

    export_parser = sub.add_parser("export-sft", help="Convert traces into SFT JSONL.")
    export_parser.add_argument("--traces", default=Path("runs"), type=Path)
    export_parser.add_argument("--output", default=Path("data/sft/tool_calls.jsonl"), type=Path)

    args = parser.parse_args()
    if args.command == "run":
        config = AgentConfig.from_env()
        model = MockCodingModel() if args.mock else OpenAICompatibleChat(config)
        result = run_task(args.repo, args.task, args.test_command, config, model)
        print(f"run_id={result.run_id}")
        print(f"outcome={result.outcome}")
        print(f"steps={result.steps}")
        print(f"workspace={result.workspace}")
        print(f"trace={result.trace_path}")
        print(f"summary={result.final_summary}")
    elif args.command == "export-sft":
        count = export_sft(args.traces, args.output)
        print(f"wrote {count} samples to {args.output}")


if __name__ == "__main__":
    main()
