# AgentTrace Sandbox

AgentTrace Sandbox is a small, dependency-light MVP for collecting coding-agent trajectories in an isolated workspace and turning successful tool calls into SFT data.

The project is designed around one practical post-training loop:

```text
coding task
  -> copy repo into an isolated run workspace
  -> let an agent choose JSON tool calls
  -> execute safe tools and tests
  -> record success/failure trace.jsonl
  -> export trace steps into SFT JSONL
```

## Why This Exists

Most coding-agent demos focus on the agent answer. This project focuses on the data exhaust that matters for post-training:

- What did the model inspect?
- Which tool did it choose?
- Did the tool call pass policy checks?
- Did tests pass?
- What diff did the task produce?
- Can the successful steps become supervised training samples?

This is an independent MVP implementation for learning and extension. It is inspired by common public coding-agent patterns such as tool loops, repository-local commands, traces, and SFT conversion, but the code here is intentionally compact and written from scratch.

## Current Features

- Copies every source repo into `runs/<run_id>/workspace` before execution.
- Keeps the original repo untouched.
- Provides safe file and test tools:
  - `list_files`
  - `read_file`
  - `grep`
  - `replace_in_file`
  - `write_file`
  - `run_tests`
  - `git_diff`
  - `finish`
- Blocks path escape, protected file edits, and dangerous commands.
- Supports OpenAI-compatible chat APIs.
- Includes a deterministic `--mock` model for local demos without an API key.
- Writes JSONL traces for every run.
- Records raw model output, retry count, step latency, final diff, and run latency.
- Marks `format_violation=true` when a model response had recoverable markdown/prose around JSON.
- Classifies outcomes such as `success`, `invalid_json`, `edit_miss`, `test_failed`, `blocked_by_policy`, and `max_steps`.
- Supports `local` and `docker` test execution backends.
- Runs JSONL task manifests for batch trajectory collection.
- Summarizes run pass rate, failure distribution, average steps, and sandbox backend usage.
- Exports successful tool calls into JSONL or Alpaca-style SFT data.
- Builds runnable unit-test benchmark tasks from offline/JSONL records.
- Builds MBPP and HumanEval-style benchmark tasks into runnable repos and manifests.
- Builds unit-test-driven function completion tasks from existing Python repos.
- Fetches GitHub PR/Issue/diff records into local JSONL.
- Builds lightweight PR/Issue repair wiki records from local JSONL.

## Quick Start

Run the built-in sample with the mock model:

```bash
python -m agenttrace_sandbox.cli run \
  --repo examples/buggy_calculator \
  --task "Fix the subtract function bug." \
  --test-command "python3 -m unittest discover -s tests" \
  --mock
```

Expected output:

```text
outcome=success
workspace=.../runs/<run_id>/workspace
trace=.../runs/<run_id>/trace.jsonl
```

Export SFT samples:

```bash
python -m agenttrace_sandbox.cli export-sft \
  --traces runs \
  --output data/sft/tool_calls.jsonl
```

Export Alpaca-style data for LLaMA-Factory-style workflows:

```bash
python -m agenttrace_sandbox.cli export-sft \
  --traces runs \
  --output data/sft/tool_calls_alpaca.json \
  --format alpaca
```

Run a task manifest:

```bash
python -m agenttrace_sandbox.cli run-manifest \
  --manifest examples/tasks.jsonl \
  --output runs/manifest_results.jsonl \
  --mock
```

Summarize collected runs:

```bash
python -m agenttrace_sandbox.cli stats --runs runs
```

Build runnable benchmark-style tasks:

```bash
python -m agenttrace_sandbox.cli build-benchmark \
  --output-dir data/benchmarks/offline \
  --limit 5
```

Build MBPP-style function tasks:

```bash
python -m agenttrace_sandbox.cli build-mbpp \
  --output-dir data/benchmarks/mbpp \
  --limit 20 \
  --dataset-source auto
```

Build HumanEval-style function tasks:

```bash
python -m agenttrace_sandbox.cli build-humaneval \
  --output-dir data/benchmarks/humaneval \
  --limit 20 \
  --dataset-source auto
```

`--dataset-source` can be `auto`, `modelscope`, `huggingface`, or `offline`.

To use Hugging Face datasets:

```bash
pip install -e ".[benchmarks]"
```

To use ModelScope datasets:

```bash
pip install -e ".[modelscope]"
export MODELSCOPE_SDK_TOKEN=your_token_here
```

`MODELSCOPE_API_TOKEN` is also accepted as an alias.

The default ModelScope dataset ids are:

```text
MBPP: OmniData/MBPP
HumanEval: openai-mirror/openai_humaneval
```

You can override them with `--modelscope-dataset`. Without optional dependencies or network access, the builders fall back to tiny built-in seed tasks so the pipeline still runs end to end.

Build unit-test-driven completion tasks from an existing Python repo:

```bash
python -m agenttrace_sandbox.cli build-unit-completion \
  --repo examples/buggy_calculator \
  --output-dir data/benchmarks/unit_completion \
  --limit 20 \
  --min-confidence 0.5 \
  --max-per-file 5 \
  --check-baseline
```

This scans `tests/test*.py`, confirms the imported functions are actually called, copies the repo, blanks the target function body, and writes a runnable `tasks.jsonl`. Supported test patterns include:

- `from math_utils import add` followed by `add(...)`
- `import math_utils` followed by `math_utils.add(...)`
- `import math_utils as mu` followed by `mu.add(...)`
- `from math_utils import add as plus` followed by `plus(...)`
- `from pkg import math_utils` followed by `math_utils.add(...)`

Module resolution supports flat modules, `src` layout, and package layout such as `math_utils.py`, `src/pkg/math_utils.py`, `pkg/math_utils.py`, and package `__init__.py` files. The blanker preserves function signatures, decorators, and docstrings, and skips targets that cannot be parsed safely. By default it targets top-level functions; class/static methods can be included with `--include-methods` when they are directly discoverable.

Before blanking a target, the builder now runs a narrow baseline check against the relevant unittest selector when it can identify one, such as `python3 -m unittest tests.test_calculator.CalculatorTests.test_add`. If that baseline test already fails in the original repo, the target is skipped. This prevents unrelated pre-existing failures from turning a correct completion into a failed trace. Local test execution also adds `src/` to `PYTHONPATH` when a repo uses src layout.

Useful builder controls:

- `--min-confidence`: minimum AST-discovery confidence to keep, default `0.5`.
- `--include-methods`: include directly discovered class/static methods.
- `--exclude-private` / `--no-exclude-private`: skip `_private` functions by default.
- `--max-per-file`: cap generated targets per source file, default `5`.
- `--check-baseline` / `--no-check-baseline`: require relevant baseline tests to pass before generating a task, default enabled.
- `--baseline-timeout`: timeout for each baseline check, default `30` seconds.

Each manifest row includes `source`, `target_file`, `target_symbol`, `test_files`, `test_selectors`, `confidence`, `original_module`, `original_import`, `test_command`, `full_test_command`, `baseline_checked`, `baseline_ok`, `baseline_command`, and `target_class` when applicable. `run-manifest` also records a simple `failure_attribution` value so unit-completion failures can be separated from baseline failures or ordinary model/task failures.

Validate the generated manifest without calling a model:

```bash
python -m agenttrace_sandbox.cli run-manifest \
  --manifest data/benchmarks/unit_completion/tasks.jsonl \
  --output runs/unit_completion_results.jsonl \
  --dry-run
```

Remove `--dry-run` when an API model is configured.

Export successful benchmark traces to SFT:

```bash
python -m agenttrace_sandbox.cli export-sft \
  --traces runs \
  --output data/sft/unit_completion_tool_calls.jsonl
```

Export only high-quality traces:

```bash
python -m agenttrace_sandbox.cli export-sft \
  --traces runs \
  --output data/sft/benchmark_tool_calls_strict.jsonl \
  --strict \
  --reject-test-edits
```

Strict export keeps only traces that succeeded, ran passing tests, had clean JSON tool calls, needed no JSON repair retries, and had no tool/policy errors. `--reject-test-edits` also removes traces that changed test files.

If the model succeeds but occasionally wraps one tool call in markdown/prose, use clean-step export:

```bash
python -m agenttrace_sandbox.cli export-sft \
  --traces runs \
  --output data/sft/benchmark_tool_calls_clean_steps.jsonl \
  --clean-steps \
  --reject-test-edits
```

This keeps only clean tool calls from successful traces, instead of rejecting the whole trace.

Summarize benchmark pass rates from a manifest result file:

```bash
python -m agenttrace_sandbox.cli stats \
  --manifest-results runs/mbpp_results.jsonl
```

Build PR/Issue repair wiki records from local JSONL:

```bash
python -m agenttrace_sandbox.cli build-pr-wiki \
  --input examples/pr_issue_pairs.jsonl \
  --output data/wiki/repair_wiki.jsonl
```

Fetch GitHub PR/Issue/diff records first:

```bash
export GITHUB_TOKEN=your_github_token

python -m agenttrace_sandbox.cli fetch-github-prs \
  --repo owner/name \
  --output data/github/pr_issue_pairs.jsonl \
  --limit 20 \
  --state closed
```

Then pass that output to `build-pr-wiki`. `GITHUB_TOKEN` is optional for public repos but recommended for rate limits.

Fetch higher-quality bug-fix PRs only:

```bash
python -m agenttrace_sandbox.cli fetch-github-prs \
  --repo django/django \
  --output data/github/django_bugfix_prs.jsonl \
  --limit 50 \
  --state closed \
  --bug-fix-only \
  --min-bug-score 2
```

The bug-fix filter uses PR title/body, linked issue title/body, changed files, and diff text. It favors keywords such as `fix`, `bug`, `regression`, `error`, `exception`, `crash`, `failing`, and common traceback names, while filtering docs-only, CI/config-only, typo/documentation, refactor/cleanup, and test-only PRs by default. `--include-docs-only` and `--include-tests-only` can relax those two structural filters when `--bug-fix-only` is enabled.

Each fetched record includes quality fields:

- `is_bug_fix`: whether the record passes the default bug-fix heuristic.
- `bug_fix_score`: numeric heuristic score for thresholding.
- `bug_fix_reasons`: short explanations for the score.
- `source_files` / `test_files`: changed files split by role.
- `docs_only` / `tests_only`: structural filters used by the scorer.

Build repair wiki records from the bug-fix JSONL:

```bash
python -m agenttrace_sandbox.cli build-pr-wiki \
  --input data/github/django_bugfix_prs.jsonl \
  --output data/wiki/django_repair_wiki.jsonl
```

When present, `bug_fix_score` and `bug_fix_reasons` are copied into wiki metadata/source context for later analysis.

Build evidence-grounded repair cards from the same PR JSONL:

```bash
python -m agenttrace_sandbox.cli build-repair-cards \
  --input data/github/django_bugfix_prs.jsonl \
  --output data/wiki/django_repair_cards.jsonl \
  --min-quality 0.6
```

Repair cards are a structured, grounded upgrade over the lightweight wiki format. Each row contains:

- `evidence`: numbered snippets such as `issue_text`, `pr_text`, `source_diff`, and `test_diff`.
- `repair_card`: grounded fields for `symptom`, `localization`, `patch_intent`, `test_oracle`, and `validation`.
- `quality`: automatic signals such as `has_test_evidence`, `evidence_coverage`, `grounding_score`, and `overall`.
- `derived_tasks`: starter targets for localization, bug explanation, repair instruction, and test-spec SFT samples.

This format is meant as an intermediate data asset: keep the structured JSONL for analysis, then derive mid-training text, instruction SFT, or agent task prompts from it.

Optionally enrich repair cards with an OpenAI-compatible model:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.deepseek.com/v1
export OPENAI_MODEL=deepseek-chat

python -m agenttrace_sandbox.cli enrich-repair-cards \
  --input data/wiki/django_repair_cards.jsonl \
  --output data/wiki/django_enriched_repair_cards.jsonl \
  --limit 20
```

The enrichment step adds `llm_repair_card` fields such as `root_cause`, `failure_condition`, `expected_behavior`, `repair_rationale`, and `edge_cases`. Each field must cite existing evidence IDs, and `llm_quality` records whether the JSON was valid, evidence IDs were valid, and grounding checks passed. API keys are read only from environment variables and should never be committed.

Run tests inside Docker instead of the host Python environment:

```bash
python -m agenttrace_sandbox.cli run-manifest \
  --manifest examples/tasks.jsonl \
  --sandbox docker \
  --docker-image python:3.11-slim \
  --mock
```

## Using An API Model

Set an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4o-mini
export AGENTTRACE_JSON_RETRIES=2
export AGENTTRACE_MAX_STEPS=8
export AGENTTRACE_SANDBOX=docker
export AGENTTRACE_DOCKER_IMAGE=python:3.11-slim
export AGENTTRACE_DOCKER_NETWORK=none
export AGENTTRACE_DOCKER_MEMORY=1g
export AGENTTRACE_DOCKER_CPUS=1
```

Then run without `--mock`:

```bash
python -m agenttrace_sandbox.cli run \
  --repo examples/buggy_calculator \
  --task "Fix the subtract function bug." \
  --test-command "python3 -m unittest discover -s tests"
```

## Trace Format

Each run creates `runs/<run_id>/trace.jsonl` with events such as:

```json
{"event": "run_started", "payload": {"task": "..."}}
{"event": "plan", "payload": {"plan": "..."}}
{"event": "tool_call", "payload": {"tool": "read_file", "arguments": {"path": "calculator.py"}, "raw_output": "{...}", "retries_used": 0, "format_violation": false, "elapsed_ms": 1.2, "result": {"ok": true}}}
{"event": "run_finished", "payload": {"outcome": "success", "diff": "...", "elapsed_ms": 45.1}}
```

When Docker mode is enabled, `run_started` also records the backend and image:

```json
{"event": "run_started", "payload": {"provider": "openai", "model": "deepseek-chat", "sandbox_backend": "docker", "docker_image": "python:3.11-slim"}}
```

Current outcome taxonomy:

```text
success
invalid_json
edit_miss
test_failed
blocked_by_policy
max_steps
finished_without_tests
incomplete
runner_error
```

## SFT Export Shape

The exporter creates records like:

```json
{
  "instruction": "Given a coding task, plan, and previous tool history, choose the next safe tool call as JSON.",
  "input": {
    "task": "Fix the subtract function bug.",
    "plan": "Inspect the relevant Python file...",
    "history": []
  },
  "output": {
    "tool": "read_file",
    "arguments": {"path": "calculator.py"},
    "reason": "Inspect the calculator implementation."
  }
}
```

## Roadmap

- Trajectory quality scoring.
- Preference pair export for DPO/RLHF-style training.
- Multi-model sampling for the same task.
- Simple dashboard for pass rate, failure reasons, and trace inspection.
- SWE-bench style task materialization and evaluation.

## Development

Run tests with the standard library:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

If you do not install the package, set `PYTHONPATH` when invoking the CLI:

```bash
PYTHONPATH=src python -m agenttrace_sandbox.cli --help
```
