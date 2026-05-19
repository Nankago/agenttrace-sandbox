# Qwen3-8B Repair Corpus A/B Test Handoff

## Goal

Compare:

- **A / base**: `/gfs/space/private/wujn/Learn/models/Qwen3-8B`
- **B / repair_lora**: base Qwen3-8B plus LoRA adapter from `outputs/qwen3_8b_repair_corpus_lora/adapter`

The LoRA adapter was trained as causal-LM midtrain / DAPT on:

```text
data/experiments/first_stage_500/corpus/repair_corpus_no_llm_strict.jsonl
```

## Current Training Result

Adapter output:

```text
outputs/qwen3_8b_repair_corpus_lora/adapter/
```

Training summary:

```text
epochs: 3
global_steps: 159
epoch 1 val_loss: 0.8248
epoch 2 val_loss: 0.7876
epoch 3 val_loss: 0.7829
```

## A/B Evaluation Outputs

Evaluation root:

```text
outputs/qwen3_8b_repair_ab_eval_parallel/
```

Expected files:

```text
outputs/qwen3_8b_repair_ab_eval_parallel/base/summary_shard_0_of_2.json
outputs/qwen3_8b_repair_ab_eval_parallel/base/summary_shard_1_of_2.json
outputs/qwen3_8b_repair_ab_eval_parallel/base/humaneval_results_shard_0_of_2.jsonl
outputs/qwen3_8b_repair_ab_eval_parallel/base/humaneval_results_shard_1_of_2.jsonl
outputs/qwen3_8b_repair_ab_eval_parallel/base/mbpp_results_shard_0_of_2.jsonl
outputs/qwen3_8b_repair_ab_eval_parallel/base/mbpp_results_shard_1_of_2.jsonl
outputs/qwen3_8b_repair_ab_eval_parallel/repair_lora/summary_shard_0_of_2.json
outputs/qwen3_8b_repair_ab_eval_parallel/repair_lora/summary_shard_1_of_2.json
outputs/qwen3_8b_repair_ab_eval_parallel/repair_lora/humaneval_results_shard_0_of_2.jsonl
outputs/qwen3_8b_repair_ab_eval_parallel/repair_lora/humaneval_results_shard_1_of_2.jsonl
outputs/qwen3_8b_repair_ab_eval_parallel/repair_lora/mbpp_results_shard_0_of_2.jsonl
outputs/qwen3_8b_repair_ab_eval_parallel/repair_lora/mbpp_results_shard_1_of_2.jsonl
outputs/qwen3_8b_repair_ab_eval_parallel/ab_summary.json
```

The eval script writes per-task JSONL incrementally. If interrupted, rerun the same shard command; completed task IDs are skipped.

## Four-GPU Background Processes

The previous single-GPU eval in `outputs/qwen3_8b_repair_ab_eval/` was stopped before switching to four GPUs. The active run is the parallel directory above.

PID files:

```text
outputs/qwen3_8b_repair_ab_eval_parallel/logs/base_shard_0.pid
outputs/qwen3_8b_repair_ab_eval_parallel/logs/base_shard_1.pid
outputs/qwen3_8b_repair_ab_eval_parallel/logs/repair_lora_shard_0.pid
outputs/qwen3_8b_repair_ab_eval_parallel/logs/repair_lora_shard_1.pid
```

Current PIDs at launch:

```text
GPU 0: base shard 0/2        PID 167126
GPU 1: base shard 1/2        PID 167128
GPU 2: repair_lora shard 0/2 PID 167130
GPU 3: repair_lora shard 1/2 PID 167132
```

Log files:

```text
outputs/qwen3_8b_repair_ab_eval_parallel/logs/base_shard_0.log
outputs/qwen3_8b_repair_ab_eval_parallel/logs/base_shard_1.log
outputs/qwen3_8b_repair_ab_eval_parallel/logs/repair_lora_shard_0.log
outputs/qwen3_8b_repair_ab_eval_parallel/logs/repair_lora_shard_1.log
```

Monitor:

```bash
nvidia-smi
ps -o pid,ppid,sid,stat,etime,cmd -p 167126,167128,167130,167132
tail -f outputs/qwen3_8b_repair_ab_eval_parallel/logs/base_shard_0.log
```

All processes were started with `setsid` + `nohup`, so they should continue after terminal or SSH disconnects.

Aggregate final results after all four shard processes finish:

```bash
python3 scripts/summarize_qwen_ab_eval.py --root outputs/qwen3_8b_repair_ab_eval_parallel
```

Check completion:

```bash
test -f outputs/qwen3_8b_repair_ab_eval_parallel/ab_summary.json && cat outputs/qwen3_8b_repair_ab_eval_parallel/ab_summary.json
```

Stop if needed:

```bash
kill "$(cat outputs/qwen3_8b_repair_ab_eval_parallel/logs/base_shard_0.pid)"
kill "$(cat outputs/qwen3_8b_repair_ab_eval_parallel/logs/base_shard_1.pid)"
kill "$(cat outputs/qwen3_8b_repair_ab_eval_parallel/logs/repair_lora_shard_0.pid)"
kill "$(cat outputs/qwen3_8b_repair_ab_eval_parallel/logs/repair_lora_shard_1.pid)"
```

## Exact Eval Commands

The four background jobs use one H800 each:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 scripts/run_qwen_ab_eval.py \
  --model /gfs/space/private/wujn/Learn/models/Qwen3-8B \
  --adapter outputs/qwen3_8b_repair_corpus_lora/adapter \
  --repair-corpus data/experiments/first_stage_500/corpus/repair_corpus_no_llm_strict.jsonl \
  --humaneval data/eval/humaneval/HumanEval.jsonl \
  --mbpp data/eval/mbpp/MBPP.jsonl \
  --output outputs/qwen3_8b_repair_ab_eval_parallel \
  --max-length 4096 \
  --max-new-tokens 256 \
  --timeout 8 \
  --only-model base \
  --num-shards 2 \
  --shard-index 0

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python3 scripts/run_qwen_ab_eval.py \
  --model /gfs/space/private/wujn/Learn/models/Qwen3-8B \
  --adapter outputs/qwen3_8b_repair_corpus_lora/adapter \
  --repair-corpus data/experiments/first_stage_500/corpus/repair_corpus_no_llm_strict.jsonl \
  --humaneval data/eval/humaneval/HumanEval.jsonl \
  --mbpp data/eval/mbpp/MBPP.jsonl \
  --output outputs/qwen3_8b_repair_ab_eval_parallel \
  --max-length 4096 \
  --max-new-tokens 256 \
  --timeout 8 \
  --only-model base \
  --num-shards 2 \
  --shard-index 1

CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 python3 scripts/run_qwen_ab_eval.py \
  --model /gfs/space/private/wujn/Learn/models/Qwen3-8B \
  --adapter outputs/qwen3_8b_repair_corpus_lora/adapter \
  --repair-corpus data/experiments/first_stage_500/corpus/repair_corpus_no_llm_strict.jsonl \
  --humaneval data/eval/humaneval/HumanEval.jsonl \
  --mbpp data/eval/mbpp/MBPP.jsonl \
  --output outputs/qwen3_8b_repair_ab_eval_parallel \
  --max-length 4096 \
  --max-new-tokens 256 \
  --timeout 8 \
  --only-model repair_lora \
  --num-shards 2 \
  --shard-index 0

CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 python3 scripts/run_qwen_ab_eval.py \
  --model /gfs/space/private/wujn/Learn/models/Qwen3-8B \
  --adapter outputs/qwen3_8b_repair_corpus_lora/adapter \
  --repair-corpus data/experiments/first_stage_500/corpus/repair_corpus_no_llm_strict.jsonl \
  --humaneval data/eval/humaneval/HumanEval.jsonl \
  --mbpp data/eval/mbpp/MBPP.jsonl \
  --output outputs/qwen3_8b_repair_ab_eval_parallel \
  --max-length 4096 \
  --max-new-tokens 256 \
  --timeout 8 \
  --only-model repair_lora \
  --num-shards 2 \
  --shard-index 1
```

## Metrics

For each model:

- `repair.val_loss`: held-out repair corpus causal-LM loss on the same deterministic 10% split seed used during training.
- `humaneval.pass_at_1`: HumanEval greedy pass@1.
- `mbpp.pass_at_1`: MBPP greedy pass@1.

Interpretation:

- Repair loss should improve for `repair_lora`.
- HumanEval/MBPP should not materially regress.
- If repair loss improves but HumanEval/MBPP regress sharply, the adapter is over-specializing.

## Notes

- This is an approximate local evaluator, not EvalPlus or SWE-bench.
- HumanEval/MBPP execution is sandbox-light: generated code is executed in a subprocess with a timeout.
- For a publishable result, rerun with EvalPlus or a standard benchmark harness after this smoke A/B test.

## MBPP Chat-Template Rerun

The raw-prompt MBPP result is suspected to be contaminated by Qwen3 prompt-format mismatch. A corrected MBPP-only rerun was started with Qwen3 chat template and `enable_thinking=False`.

Evaluation root:

```text
outputs/qwen3_8b_repair_mbpp_chat_eval_parallel/
```

Current PIDs at launch:

```text
GPU 0: base shard 0/2        PID 203429
GPU 1: base shard 1/2        PID 203431
GPU 2: repair_lora shard 0/2 PID 203433
GPU 3: repair_lora shard 1/2 PID 203435
```

Monitor:

```bash
nvidia-smi
ps -o pid,ppid,sid,stat,etime,cmd -p 203429,203431,203433,203435
tail -f outputs/qwen3_8b_repair_mbpp_chat_eval_parallel/logs/base_shard_0.log
```

Aggregate after all four processes finish:

```bash
python3 scripts/summarize_qwen_ab_eval.py --root outputs/qwen3_8b_repair_mbpp_chat_eval_parallel
```

The rerun command uses:

```text
--skip-humaneval
--mbpp-prompt-style chat
--max-new-tokens 384
```

## Second-Stage 2k PR Experiment

A second-stage pipeline was started to build a higher-quality 2k PR repair corpus, train a gentler LoRA, and run A/B evaluation.

Pipeline PID:

```text
229475
```

Pipeline log:

```text
outputs/second_stage_2k_pipeline/pipeline.log
```

Main outputs:

```text
data/experiments/second_stage_2k/
outputs/qwen3_8b_repair_corpus_lora_2k_e1_lr5e5/
outputs/qwen3_8b_repair_2k_ab_eval_parallel/
```

Training settings:

```text
epochs: 1
lr: 5e-5
LoRA target: Qwen3-8B
data: data/experiments/second_stage_2k/corpus/repair_corpus_semantic.jsonl
```

Monitor:

```bash
ps -o pid,ppid,sid,stat,etime,cmd -p 229475
tail -f outputs/second_stage_2k_pipeline/pipeline.log
```

When it reaches evaluation, per-shard eval logs are under:

```text
outputs/qwen3_8b_repair_2k_ab_eval_parallel/logs/
```

Final summary, after completion:

```bash
cat outputs/qwen3_8b_repair_2k_ab_eval_parallel/ab_summary.json
```
