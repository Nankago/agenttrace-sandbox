# Qwen Repair LoRA A/B Evaluation Diagnosis

## Observed Result

Four-GPU raw-prompt A/B evaluation finished under:

```text
outputs/qwen3_8b_repair_ab_eval_parallel/
```

Aggregated result:

```text
base:
  repair val_loss: 1.5629
  HumanEval pass@1: 22 / 164 = 13.41%
  MBPP pass@1:      46 / 974 = 4.72%

repair_lora:
  repair val_loss: 0.7829
  HumanEval pass@1: 24 / 164 = 14.63%
  MBPP pass@1:       8 / 974 = 0.82%
```

## Main Diagnosis

The MBPP drop is mostly an evaluation-format and output-format failure, amplified by the LoRA's domain drift. It should not be read as a clean measurement that the repair LoRA destroyed code ability by that full amount.

Evidence:

```text
Raw MBPP error categories:

base:
  syntax: 745
  name:   121
  indent: 51
  ok:     46

repair_lora:
  name:   867
  indent: 73
  syntax: 18
  ok:      8
```

The LoRA generated far fewer usable function definitions and much more prompt/comment repetition:

```text
base:
  repeated "Return code only" prompt: 4 / 974
  contains "Do not write any explanation": 134 / 974
  contains def: 158 / 974

repair_lora:
  repeated "Return code only" prompt: 107 / 974
  contains "Do not write any explanation": 667 / 974
  contains def: 91 / 974
```

Post-hoc improved code extraction, without regenerating, only partly helped:

```text
base:        49 / 974 = 5.03%
repair_lora: 19 / 974 = 1.95%
```

So extraction was part of the issue, but many LoRA completions still did not contain executable code.

## Chat Template Check

The local Qwen3-8B tokenizer has a chat template. The original MBPP eval used a raw prompt, which is not the right interface for this model family.

A 20-example MBPP smoke test using chat template plus `enable_thinking=False` produced:

```text
base:        13 / 20
repair_lora: 12 / 20
```

This strongly suggests that the severe `0.82%` MBPP score is mostly a harness artifact plus LoRA output-format sensitivity, not purely a true capability collapse.

## Training-Side Cause

The LoRA was trained as causal-LM continued pretraining on 465 repair-corpus records. The corpus is mostly PR narrative, evidence blocks, summaries, and diffs, not standalone coding-problem solutions.

Corpus signals:

```text
rows: 465
avg chars: 7677
Repository:/PR:/Evidence:/Symptom:/Root Cause:/Patch Intent:/Test Oracle: 465 / 465
contains diff headers: 465 / 465
contains markdown fences: 257 / 465
code-ish keyword ratio: 0.0381
```

That makes the LoRA better at the repair-corpus distribution, but also biases generation toward prose, copied headings, repeated instructions, and PR-analysis style text.

## Script Fix

`scripts/run_qwen_ab_eval.py` now supports:

```text
--mbpp-prompt-style raw
--mbpp-prompt-style chat
```

It also uses a stronger `extract_code()` that:

- dedents fenced code
- starts from the first top-level `def` or `class` when explanations precede code
- strips leading comment-only prompt echoes when no code block is found

## Recommended Next Rerun

For Qwen3, rerun MBPP with:

```bash
--mbpp-prompt-style chat
--max-new-tokens 384
```

Keep HumanEval as prefix-completion style unless separately switching to an instruction-style HumanEval harness.

## Next Training Fix

For the next ablation, do not treat this LoRA as a final-quality midtrain recipe. Try a gentler run:

```text
epochs: 1
lr: 2e-5 to 5e-5
LoRA rank: 8 or 16
mix ratio: add general code data or MBPP-style code snippets
```

Also consider adding a code-only or patch-only corpus variant to isolate whether the prose-heavy repair record is causing the output-format drift.
