# Real API Trace Notes

This note summarizes observations from running AgentTrace Sandbox with a real OpenAI-compatible model endpoint. It is intentionally written as an engineering record: what happened, why it matters, and how it shaped the current implementation.

No API keys, tokens, or private endpoint secrets are recorded here.

## Experiment Setup

- Model endpoint type: OpenAI-compatible chat completions
- Test model used during local experiments: `deepseek-chat`
- Task: fix the `subtract` bug in `examples/buggy_calculator`
- Test command: `python3 -m unittest discover -s tests`
- Sandbox backend during these runs: `local`
- Number of real API runs inspected: 8 total
  - 1 initial smoke test
  - 5 repeated runs before prompt tightening
  - 1 smoke test after prompt tightening
  - 1 MBPP-style benchmark-to-SFT smoke test

## What Worked

The real model successfully completed the toy repair task.

Observed successful tool chains included:

```text
list_files -> read_file -> replace_in_file -> run_tests -> finish
grep -> read_file -> replace_in_file -> run_tests -> finish
```

All repeated real API runs reached `success`, and tests passed after the code edit.

This validates the core MVP loop:

```text
real model output
  -> JSON tool-call extraction
  -> isolated workspace edit
  -> test execution
  -> trace recording
  -> SFT-ready data export
```

The MBPP-style benchmark smoke test also completed successfully:

```text
build-mbpp
  -> run-manifest with deepseek-chat
  -> trace.jsonl
  -> export-sft
```

That run produced 1 successful trace and 3 SFT tool-call samples, covering `read_file`, `replace_in_file`, and `run_tests`.

## Issues Observed

### 1. Markdown-Wrapped Tool Calls

In early real API runs, the model often returned JSON inside markdown fences, sometimes with short natural-language lead-in text.

Example shape:

````text
I will inspect the file first.

```json
{"tool": "read_file", "arguments": {"path": "calculator.py"}, "reason": "..."}
```
````

The existing parser could still extract the JSON object, so execution succeeded. However, this is not ideal for trajectory collection because the raw response does not strictly match the desired tool-call protocol.

Why it matters:

- A coding-agent runtime may tolerate recoverable formatting.
- A post-training dataset should be able to distinguish clean protocol-following samples from recovered samples.
- Without tracking this, SFT data can silently include examples where the model was not actually following the required format.

Implemented response:

- Added stricter actor prompt instructions: raw JSON only, no markdown, no prose.
- Added `format_violation` to every tool-call trace event.
- Added `format_violation_rate` to `stats`.
- Included `format_violation` in exported SFT metadata.

### 2. Planner Suggested Out-of-Scope Actions

In the initial smoke test, the model's plan included actions such as committing the change.

The runtime does not expose a commit tool. More importantly, commit/PR/package-install steps are not part of the minimal task loop.

Why it matters:

- Plans can shape later tool choices.
- If plans mention unavailable actions, the agent may drift into invalid tool requests or unnecessary work.
- For training data, the plan should reflect the actual tool/action space.

Implemented response:

- Tightened the planner system prompt.
- The planner is now told to use only available file/test tools.
- It is told not to propose commits, pull requests, package installs, or extra test rewrites unless explicitly requested.

### 3. Extra Test Rewrites

One real API run fixed the code correctly but then edited the tests to add extra cases.

The final result was still valid, but it was not the smallest change for the task.

Why it matters:

- For benchmark-style repair tasks, minimal diffs are usually preferred.
- Extra edits increase risk and make trajectories noisier.
- If used for SFT, such traces may teach the model to make broad edits when a narrow fix is enough.

Implemented response:

- Tightened actor prompt: prefer minimal code edits.
- Added explicit instruction not to modify tests unless the task asks for tests or existing tests are clearly wrong.
- After the prompt update, a real API smoke run completed in 5 steps and only changed `calculator.py`.

### 4. Need To Track Model Metadata

Early traces recorded the task, workspace, and sandbox backend, but not the model identity.

Why it matters:

- Trajectory quality depends strongly on the model.
- Multi-model experiments need attribution for later analysis.
- Post-training data should preserve provenance.

Implemented response:

- `run_started` now records:
  - `provider`
  - `model`
  - `temperature`
  - `sandbox_backend`
  - `docker_image` when applicable

### 5. API Key Handling During Manual Tests

Manual command-line tests can accidentally echo secrets if the key is typed into a normal prompt or embedded directly in commands.

Why it matters:

- Secrets should never be committed or written into trace files.
- Developer ergonomics should make the safe path easy.

Current practice:

- Real API smoke tests use environment variables.
- Keys are not written to `.env`, source files, README, docs, traces, or committed files.
- For manual one-off tests, prefer silent shell input or pre-set environment variables.

Future improvement:

- Add `.env.example` documenting variable names without secrets.
- Add a small `agenttrace doctor` command to check API configuration without printing secrets.

## Current Trace Improvements

The real API observations led to these trace fields:

```json
{
  "raw_output": "...",
  "parse_error": "",
  "retries_used": 0,
  "format_violation": false,
  "elapsed_ms": 12.34,
  "result": {
    "ok": true,
    "error_type": ""
  }
}
```

And run-level metadata:

```json
{
  "provider": "openai",
  "model": "deepseek-chat",
  "temperature": 0.1,
  "sandbox_backend": "local"
}
```

## Innovation Points From These Findings

These observations support a clearer project positioning:

### Protocol Compliance Tracking

The platform does not merely execute agent outputs. It measures whether the model followed the tool-call protocol cleanly or required recovery.

Useful metrics:

- valid JSON rate
- format violation rate
- retry rate
- unknown tool rate
- policy block rate

### Safe Recovery vs. Clean Training Data

At runtime, recoverable outputs can be tolerated for robustness.

For training, the same traces can be filtered:

```text
strict SFT set: format_violation=false and result.ok=true
recovery set: format_violation=true but parse succeeded
failure set: parse_error or policy/test failure
```

This creates a richer data pipeline than simply saving successful actions.

### Minimal-Diff Agent Behavior

The real model sometimes wanted to improve tests even when the task only required a code fix.

This motivates a quality dimension beyond pass/fail:

- Was the diff minimal?
- Did the agent edit tests unnecessarily?
- Did it use the smallest sufficient tool sequence?
- Did it stop after validation?

### Model-Provenance-Aware Trace Collection

Recording model/provider/temperature makes it possible to compare trajectory quality across models.

Potential future analysis:

```text
model A: lower format violations but more test failures
model B: higher success but more unnecessary edits
model C: better minimal-diff behavior
```

### Sandbox-First Real API Evaluation

Real API calls can produce unpredictable file edits and command choices. Combining model sampling with local/Docker sandboxing is central to making trajectory collection safe and repeatable.

## Follow-Up Ideas

- Add `agenttrace doctor` to validate API config and Docker availability.
- Add strict SFT export mode:

```bash
agenttrace export-sft --strict
```

that only exports clean protocol-following successful tool calls.

- Add trajectory quality scoring:

```text
success + clean JSON + tests passed + minimal diff + no unnecessary test edit
```

- Add DPO pair generation:

```text
chosen: clean/minimal successful trace
rejected: recovered/noisy/over-editing trace
```

- Add model comparison stats grouped by `model`.

## ModelScope Dataset Loading Notes

When adding ModelScope as a benchmark source, two practical issues appeared:

- Installing `modelscope` alone was not enough in the local Python environment; importing `MsDataset` also required runtime packages such as `numpy`, `datasets`, and `addict`.
- ModelScope dataset split names can differ from the Hugging Face mirror. For example, `openai-mirror/openai_humaneval` exposes the expected `test` split, while `OmniData/MBPP` exposes `train` rather than `test`.

Implemented response:

- Added a `modelscope` optional dependency group that includes the required runtime packages.
- Added a ModelScope loader that uses `MODELSCOPE_SDK_TOKEN` or `MODELSCOPE_API_TOKEN` from the environment without printing or storing the token.
- Added split fallback logic for ModelScope datasets: requested split, then `train`, `test`, `validation`, and `dev`.

## Small Benchmark Smoke Results

After adding MBPP/HumanEval builders and ModelScope loading, a small real-API benchmark smoke test was run:

```text
MBPP sample size: 3
HumanEval sample size: 3
Model endpoint type: OpenAI-compatible chat completions
Model used: deepseek-chat
Sandbox backend: local
Max steps: 10
```

Observed results:

```text
MBPP:      3/3 success, pass_rate=1.0, avg_steps=6.00, format_violation_rate=0.1111
HumanEval: 3/3 success, pass_rate=1.0, avg_steps=4.67, format_violation_rate=0.2857
```

SFT export counts:

```text
MBPP all successful tool calls:         14
MBPP strict whole-trace samples:         0
MBPP clean-step samples:                12
HumanEval all successful tool calls:    11
HumanEval strict whole-trace samples:    0
HumanEval clean-step samples:            7
```

Interpretation:

- Functional pass rate was good on this tiny sample.
- Strict whole-trace export was zero because every successful trace had at least one recoverable format violation.
- Clean-step export is currently more useful for building SFT data from real model runs, while strict export is useful as a protocol-compliance target.
