from __future__ import annotations

import argparse
from pathlib import Path

from agenttrace_sandbox.config import AgentConfig
from agenttrace_sandbox.llm import MockCodingModel, OpenAICompatibleChat
from agenttrace_sandbox.manifest import run_manifest
from agenttrace_sandbox.runner import run_task
from agenttrace_sandbox.sft_export import export_sft
from agenttrace_sandbox.stats import compute_run_stats


def add_sandbox_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sandbox", choices=["local", "docker"], help="Execution backend for test commands.")
    parser.add_argument("--docker-image", help="Docker image used when --sandbox docker is selected.")
    parser.add_argument("--docker-network", help="Docker network mode, defaults to none.")
    parser.add_argument("--docker-memory", help="Docker memory limit, defaults to 1g.")
    parser.add_argument("--docker-cpus", help="Docker CPU limit, defaults to 1.")


def apply_sandbox_args(config: AgentConfig, args: argparse.Namespace) -> AgentConfig:
    return AgentConfig(
        provider=config.provider,
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=config.temperature,
        max_steps=config.max_steps,
        json_retries=config.json_retries,
        command_timeout=config.command_timeout,
        sandbox_backend=args.sandbox or config.sandbox_backend,
        docker_image=args.docker_image or config.docker_image,
        docker_network=args.docker_network or config.docker_network,
        docker_memory=args.docker_memory or config.docker_memory,
        docker_cpus=args.docker_cpus or config.docker_cpus,
        runs_dir=config.runs_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="agenttrace", description="Safe trajectory collection for coding-agent post-training.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run one coding-agent task in an isolated workspace.")
    run_parser.add_argument("--repo", required=True, type=Path)
    run_parser.add_argument("--task", required=True)
    run_parser.add_argument("--test-command", default="python3 -m unittest discover -s tests")
    run_parser.add_argument("--mock", action="store_true", help="Use deterministic local model instead of an API.")
    add_sandbox_args(run_parser)

    manifest_parser = sub.add_parser("run-manifest", help="Run a JSONL task manifest and collect traces.")
    manifest_parser.add_argument("--manifest", required=True, type=Path)
    manifest_parser.add_argument("--output", default=Path("runs/manifest_results.jsonl"), type=Path)
    manifest_parser.add_argument("--limit", type=int)
    manifest_parser.add_argument("--dry-run", action="store_true")
    manifest_parser.add_argument("--mock", action="store_true", help="Use deterministic local model instead of an API.")
    add_sandbox_args(manifest_parser)

    export_parser = sub.add_parser("export-sft", help="Convert traces into SFT JSONL.")
    export_parser.add_argument("--traces", default=Path("runs"), type=Path)
    export_parser.add_argument("--output", default=Path("data/sft/tool_calls.jsonl"), type=Path)
    export_parser.add_argument("--format", choices=["jsonl", "alpaca"], default="jsonl")

    stats_parser = sub.add_parser("stats", help="Summarize run traces.")
    stats_parser.add_argument("--runs", default=Path("runs"), type=Path)

    args = parser.parse_args()
    if args.command == "run":
        config = apply_sandbox_args(AgentConfig.from_env(), args)
        model = MockCodingModel() if args.mock else OpenAICompatibleChat(config)
        result = run_task(args.repo, args.task, args.test_command, config, model)
        print(f"run_id={result.run_id}")
        print(f"outcome={result.outcome}")
        print(f"steps={result.steps}")
        print(f"workspace={result.workspace}")
        print(f"trace={result.trace_path}")
        print(f"summary={result.final_summary}")
    elif args.command == "run-manifest":
        config = apply_sandbox_args(AgentConfig.from_env(), args)
        model = MockCodingModel() if args.mock else OpenAICompatibleChat(config)
        summary = run_manifest(args.manifest, config, model, args.output, limit=args.limit, dry_run=args.dry_run)
        print(f"total={summary.total}")
        print(f"ran={summary.ran}")
        print(f"skipped={summary.skipped}")
        print(f"outcomes={summary.outcomes}")
        print(f"output={summary.output_path}")
    elif args.command == "export-sft":
        count = export_sft(args.traces, args.output, output_format=args.format)
        print(f"wrote {count} samples to {args.output}")
    elif args.command == "stats":
        print(compute_run_stats(args.runs).render())


if __name__ == "__main__":
    main()
