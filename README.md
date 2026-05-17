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
- Classifies outcomes such as `success`, `invalid_json`, `edit_miss`, `test_failed`, `blocked_by_policy`, and `max_steps`.
- Supports `local` and `docker` test execution backends.
- Runs JSONL task manifests for batch trajectory collection.
- Summarizes run pass rate, failure distribution, average steps, and sandbox backend usage.
- Exports successful tool calls into JSONL or Alpaca-style SFT data.

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
{"event": "tool_call", "payload": {"tool": "read_file", "arguments": {"path": "calculator.py"}, "raw_output": "{...}", "retries_used": 0, "elapsed_ms": 1.2, "result": {"ok": true}}}
{"event": "run_finished", "payload": {"outcome": "success", "diff": "...", "elapsed_ms": 45.1}}
```

When Docker mode is enabled, `run_started` also records the backend and image:

```json
{"event": "run_started", "payload": {"sandbox_backend": "docker", "docker_image": "python:3.11-slim"}}
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
