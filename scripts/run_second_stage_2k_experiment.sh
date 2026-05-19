#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

TOKEN_FILE="${GITHUB_TOKEN_FILE:-/tmp/agenttrace_github_token}"
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  if [[ ! -f "$TOKEN_FILE" ]]; then
    echo "Missing GITHUB_TOKEN and token file: $TOKEN_FILE" >&2
    exit 1
  fi
  export GITHUB_TOKEN
  GITHUB_TOKEN="$(cat "$TOKEN_FILE")"
fi

MODEL="${MODEL:-/gfs/space/private/wujn/Learn/models/Qwen3-8B}"
ROOT="${ROOT:-data/experiments/second_stage_2k}"
TRAIN_OUT="${TRAIN_OUT:-outputs/qwen3_8b_repair_corpus_lora_2k_e1_lr5e5}"
EVAL_OUT="${EVAL_OUT:-outputs/qwen3_8b_repair_2k_ab_eval_parallel}"
LOG_ROOT="${LOG_ROOT:-outputs/second_stage_2k_pipeline}"
mkdir -p "$LOG_ROOT" "$EVAL_OUT/logs"

echo "[1/4] Build 2k repair corpus"
python3 scripts/build_pr_repair_experiment.py \
  --output-root "$ROOT" \
  --target 2000 \
  --max-prs-per-repo 500 \
  --min-bug-score 3.0 \
  --min-card-quality 0.85 \
  --workers 8 \
  2>&1 | tee "$LOG_ROOT/build.log"

echo "[2/4] Train Qwen3-8B repair LoRA on 2k corpus"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 scripts/train_repair_corpus_lora.py \
  --model "$MODEL" \
  --data "$ROOT/corpus/repair_corpus_semantic.jsonl" \
  --output "$TRAIN_OUT" \
  --max-length 4096 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 5e-5 \
  2>&1 | tee "$LOG_ROOT/train.log"

echo "[3/4] Run four-way A/B eval"
for spec in "0 base 0" "1 base 1" "2 repair_lora 0" "3 repair_lora 1"; do
  set -- $spec
  gpu=$1
  model_name=$2
  shard=$3
  log="$EVAL_OUT/logs/${model_name}_shard_${shard}.log"
  pid="$EVAL_OUT/logs/${model_name}_shard_${shard}.pid"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python3 scripts/run_qwen_ab_eval.py \
    --model "$MODEL" \
    --adapter "$TRAIN_OUT/adapter" \
    --repair-corpus "$ROOT/corpus/repair_corpus_semantic.jsonl" \
    --humaneval data/eval/humaneval/HumanEval.jsonl \
    --mbpp data/eval/mbpp/MBPP.jsonl \
    --output "$EVAL_OUT" \
    --max-length 4096 \
    --max-new-tokens 384 \
    --timeout 8 \
    --mbpp-prompt-style chat \
    --only-model "$model_name" \
    --num-shards 2 \
    --shard-index "$shard" \
    > "$log" 2>&1 &
  echo $! > "$pid"
  echo "started gpu=$gpu model=$model_name shard=$shard pid=$(cat "$pid")"
done

wait

echo "[4/4] Summarize eval"
python3 scripts/summarize_qwen_ab_eval.py --root "$EVAL_OUT" 2>&1 | tee "$LOG_ROOT/eval_summary.log"

echo "done"
