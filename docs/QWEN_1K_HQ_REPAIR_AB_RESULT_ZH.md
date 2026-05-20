# Qwen3-8B 1k High-Quality PR Repair Corpus A/B Result

## Experiment

This experiment validates whether a high-quality GitHub PR-derived repair corpus provides a useful midtrain signal for Qwen3-8B.

Setup:

```text
base model: /gfs/space/private/wujn/Learn/models/Qwen3-8B
method: LoRA
epochs: 1
lr: 5e-5
max_length: 4096
batch_size: 1
grad_accum: 8
train corpus: data/experiments/second_stage_1k_hq/corpus/repair_corpus_semantic.jsonl
adapter: outputs/qwen3_8b_repair_corpus_lora_1k_hq_e1_lr5e5/adapter
```

The data and output artifacts are local experiment artifacts and are not stored in git.

## Data Summary

The corpus was selected from cached second-stage GitHub PR candidates using the existing bug-fix and repair-card quality filters.

```text
selected_prs: 1000
repair_cards: 1000
corpus_records: 1000
sft_records: 4000
quality_avg: 0.97617
has_test_evidence: 1000 / 1000
has_source_patch: 1000 / 1000
```

Qwen3-8B tokenizer statistics:

```text
records:        1000
total_tokens:   2,094,471
avg_tokens:     2,094 / record
median_tokens:  1,961 / record
p90_tokens:     2,926
p95_tokens:     3,353
p99_tokens:     4,491
max_tokens:     5,896
over_4096:      19 records
```

Only 19 records exceed the 4096-token training limit, so truncation is limited.

## Training Result

Training completed successfully.

```text
global_steps: 113
repair_lora val_loss: 0.8371651613712311
```

Repair-domain validation loss comparison:

```text
base repair val_loss:        1.4605111819505692
repair_lora repair val_loss: 0.8371651613712311
delta:                       -0.6233460205793381
```

This is the main positive signal: the model fits held-out repair-card style examples much better after LoRA training.

## A/B Evaluation

The same local evaluator was run for base Qwen3-8B and the repair LoRA:

```text
eval root: outputs/qwen3_8b_repair_1k_hq_ab_eval_parallel
repair validation split: deterministic 10% split
HumanEval: local greedy pass@1
MBPP: local greedy pass@1 with Qwen chat template and enable_thinking=False
```

Final results:

```text
base HumanEval:        17 / 164 = 10.37%
repair_lora HumanEval: 19 / 164 = 11.59%
delta HumanEval:       +1.22 percentage points

base MBPP:             662 / 974 = 67.97%
repair_lora MBPP:      665 / 974 = 68.28%
delta MBPP:            +0.31 percentage points
```

Summary JSON:

```text
outputs/qwen3_8b_repair_1k_hq_ab_eval_parallel/ab_summary.json
```

## Interpretation

The 1k high-quality PR repair corpus is a positive scale point:

- Repair validation loss drops clearly from `1.4605` to `0.8372`.
- HumanEval does not regress and improves slightly from `10.37%` to `11.59%`.
- MBPP does not regress and improves slightly from `67.97%` to `68.28%`.

The result supports continuing with LoRA-style ablations and larger high-quality PR corpora. It does not by itself justify full fine-tuning, because the corpus is still small at roughly 2.09M tokens and the code-generation eval is local smoke-style rather than EvalPlus or SWE-bench.

## Position In Scale Curve

Current scale points:

```text
500 first-stage:
repair loss: 1.5629 -> 0.7829
HumanEval:   13.41% -> 14.63%
MBPP chat:   67.97% -> 67.45%

1000 high-quality:
repair loss: 1.4605 -> 0.8372
HumanEval:   10.37% -> 11.59%
MBPP chat:   67.97% -> 68.28%

1470 second-stage:
repair loss: 1.4787 -> 0.8043
HumanEval:   10.37% -> 13.41%
MBPP chat:   67.97% -> 67.76%
```

The next useful check is the 5k high-quality PR repair corpus under the same recipe. If 5k lowers repair loss further without HumanEval/MBPP regression, the data pipeline has a much stronger scaling argument.

