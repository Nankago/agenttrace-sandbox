from __future__ import annotations

import argparse
from pathlib import Path

from agenttrace_sandbox.config import AgentConfig
from agenttrace_sandbox.data_builders import (
    build_benchmark_tasks,
    build_humaneval_tasks,
    build_mbpp_tasks,
    build_pr_wiki,
    build_unit_completion_tasks,
    fetch_github_prs,
)
from agenttrace_sandbox.llm import MockCodingModel, OpenAICompatibleChat
from agenttrace_sandbox.manifest import run_manifest
from agenttrace_sandbox.runner import run_task
from agenttrace_sandbox.sft_export import export_sft
from agenttrace_sandbox.stats import compute_manifest_stats, compute_run_stats


def add_sandbox_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sandbox", choices=["local", "docker"], help="Execution backend for test commands.")
    parser.add_argument("--docker-image", help="Docker image used when --sandbox docker is selected.")
    parser.add_argument("--docker-network", help="Docker network mode, defaults to none.")
    parser.add_argument("--docker-memory", help="Docker memory limit, defaults to 1g.")
    parser.add_argument("--docker-cpus", help="Docker CPU limit, defaults to 1.")


def config_from_args(args: argparse.Namespace) -> AgentConfig:
    config = AgentConfig.from_env()
    return config.with_overrides(
        sandbox_backend=getattr(args, "sandbox", None),
        docker_image=getattr(args, "docker_image", None),
        docker_network=getattr(args, "docker_network", None),
        docker_memory=getattr(args, "docker_memory", None),
        docker_cpus=getattr(args, "docker_cpus", None),
    )


def model_from_args(args: argparse.Namespace, config: AgentConfig):
    return MockCodingModel() if getattr(args, "mock", False) else OpenAICompatibleChat(config)


def print_result(**items) -> None:
    for key, value in items.items():
        print(f"{key}={value}")


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
    export_parser.add_argument("--strict", action="store_true", help="Only export clean successful traces with passing tests.")
    export_parser.add_argument("--clean-steps", action="store_true", help="Export clean tool calls from successful traces, even if other steps were noisy.")
    export_parser.add_argument("--reject-test-edits", action="store_true", help="With --strict, reject traces that edit tests.")

    stats_parser = sub.add_parser("stats", help="Summarize run traces.")
    stats_parser.add_argument("--runs", default=Path("runs"), type=Path)
    stats_parser.add_argument("--manifest-results", type=Path, help="Summarize a run-manifest result JSONL by source.")

    benchmark_parser = sub.add_parser("build-benchmark", help="Build runnable unit-test repair tasks.")
    benchmark_parser.add_argument("--output-dir", default=Path("data/benchmarks/offline"), type=Path)
    benchmark_parser.add_argument("--limit", type=int, default=5)
    benchmark_parser.add_argument("--input", type=Path, help="Optional JSONL benchmark records.")

    unit_parser = sub.add_parser("build-unit-completion", help="Build function-completion tasks by blanking functions imported by tests.")
    unit_parser.add_argument("--repo", required=True, type=Path)
    unit_parser.add_argument("--output-dir", default=Path("data/benchmarks/unit_completion"), type=Path)
    unit_parser.add_argument("--limit", type=int, default=20)
    unit_parser.add_argument("--tests-dir", default="tests")
    unit_parser.add_argument("--test-command", default="python3 -m unittest discover -s tests")

    mbpp_parser = sub.add_parser("build-mbpp", help="Build runnable MBPP-style function implementation tasks.")
    mbpp_parser.add_argument("--output-dir", default=Path("data/benchmarks/mbpp"), type=Path)
    mbpp_parser.add_argument("--limit", type=int, default=20)
    mbpp_parser.add_argument("--split", default="test")
    mbpp_parser.add_argument("--input", type=Path, help="Optional local MBPP JSONL records.")
    mbpp_parser.add_argument("--dataset-source", choices=["auto", "modelscope", "huggingface", "offline"], default="auto")
    mbpp_parser.add_argument("--modelscope-dataset", default="OmniData/MBPP")

    humaneval_parser = sub.add_parser("build-humaneval", help="Build runnable HumanEval-style function implementation tasks.")
    humaneval_parser.add_argument("--output-dir", default=Path("data/benchmarks/humaneval"), type=Path)
    humaneval_parser.add_argument("--limit", type=int, default=20)
    humaneval_parser.add_argument("--split", default="test")
    humaneval_parser.add_argument("--input", type=Path, help="Optional local HumanEval JSONL records.")
    humaneval_parser.add_argument("--dataset-source", choices=["auto", "modelscope", "huggingface", "offline"], default="auto")
    humaneval_parser.add_argument("--modelscope-dataset", default="openai-mirror/openai_humaneval")

    wiki_parser = sub.add_parser("build-pr-wiki", help="Build repair wiki records from PR/Issue JSONL.")
    wiki_parser.add_argument("--input", required=True, type=Path)
    wiki_parser.add_argument("--output", default=Path("data/wiki/repair_wiki.jsonl"), type=Path)

    github_parser = sub.add_parser("fetch-github-prs", help="Fetch GitHub PR/Issue/diff records into local JSONL.")
    github_parser.add_argument("--repo", required=True, help="Repository in owner/name form.")
    github_parser.add_argument("--output", default=Path("data/github/pr_issue_pairs.jsonl"), type=Path)
    github_parser.add_argument("--limit", type=int, default=20)
    github_parser.add_argument("--state", choices=["open", "closed", "all"], default="closed")
    github_parser.add_argument("--bug-fix-only", action="store_true", help="Only write PRs that look like bug fixes.")
    github_parser.add_argument("--min-bug-score", type=float, default=2, help="Minimum bug-fix score used with --bug-fix-only.")
    github_parser.add_argument("--include-docs-only", action="store_true", help="Allow docs/CI-only PRs through bug-fix filtering.")
    github_parser.add_argument("--include-tests-only", action="store_true", help="Allow test-only PRs through bug-fix filtering.")

    args = parser.parse_args()
    if args.command == "run":
        config = config_from_args(args)
        model = model_from_args(args, config)
        result = run_task(args.repo, args.task, args.test_command, config, model)
        print_result(
            run_id=result.run_id,
            outcome=result.outcome,
            steps=result.steps,
            workspace=result.workspace,
            trace=result.trace_path,
            summary=result.final_summary,
        )
    elif args.command == "run-manifest":
        config = config_from_args(args)
        model = model_from_args(args, config)
        summary = run_manifest(args.manifest, config, model, args.output, limit=args.limit, dry_run=args.dry_run)
        print_result(total=summary.total, ran=summary.ran, skipped=summary.skipped, outcomes=summary.outcomes, output=summary.output_path)
    elif args.command == "export-sft":
        count = export_sft(
            args.traces,
            args.output,
            output_format=args.format,
            strict=args.strict,
            clean_steps=args.clean_steps,
            reject_test_edits=args.reject_test_edits,
        )
        print(f"wrote {count} samples to {args.output}")
    elif args.command == "stats":
        if args.manifest_results:
            print(compute_manifest_stats(args.manifest_results).render())
        else:
            print(compute_run_stats(args.runs).render())
    elif args.command == "build-benchmark":
        print(build_benchmark_tasks(args.output_dir, limit=args.limit, source_path=args.input).render())
    elif args.command == "build-unit-completion":
        print(
            build_unit_completion_tasks(
                args.repo,
                args.output_dir,
                limit=args.limit,
                tests_dir=args.tests_dir,
                test_command=args.test_command,
            ).render()
        )
    elif args.command == "build-mbpp":
        print(
            build_mbpp_tasks(
                args.output_dir,
                limit=args.limit,
                split=args.split,
                source_path=args.input,
                dataset_source=args.dataset_source,
                modelscope_dataset=args.modelscope_dataset,
            ).render()
        )
    elif args.command == "build-humaneval":
        print(
            build_humaneval_tasks(
                args.output_dir,
                limit=args.limit,
                split=args.split,
                source_path=args.input,
                dataset_source=args.dataset_source,
                modelscope_dataset=args.modelscope_dataset,
            ).render()
        )
    elif args.command == "build-pr-wiki":
        print(build_pr_wiki(args.input, args.output).render())
    elif args.command == "fetch-github-prs":
        print(
            fetch_github_prs(
                args.repo,
                args.output,
                limit=args.limit,
                state=args.state,
                bug_fix_only=args.bug_fix_only,
                min_bug_score=args.min_bug_score,
                include_docs_only=args.include_docs_only,
                include_tests_only=args.include_tests_only,
            ).render()
        )


if __name__ == "__main__":
    main()
