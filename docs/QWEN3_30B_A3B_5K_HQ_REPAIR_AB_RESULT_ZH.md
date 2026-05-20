# Qwen3-30B-A3B 5k High-Quality PR Repair Corpus A/B Result

## 实验目标

本实验用于验证：在同一批 5k high-quality GitHub PR repair corpus 上，Qwen3-30B-A3B-Instruct-2507 经过 LoRA 修复语料训练后，repair-domain loss 是否下降，并检查 HumanEval / MBPP 是否出现通用代码能力退化。

训练配置保持固定：

```text
base model: /gfs/space/private/xb/model/Qwen3-30B-A3B-Instruct-2507
method: LoRA
epochs: 1
lr: 5e-5
max_length: 4096
batch_size: 1
grad_accum: 8
train corpus: data/experiments/second_stage_5k_hq/corpus/repair_corpus_semantic.jsonl
adapter: outputs/qwen3_30b_a3b_side/lora_5k_hq_e1_lr5e5/adapter
```

`data/` 和 `outputs/` 是本地实验产物，不随 git 保存。

## 数据规模

5k corpus 构造结果：

```text
records_total_before_selection: 7230
selected_prs: 6455
repair_cards: 5000
corpus_records: 5000
sft_records: 20000
quality_avg: 0.9507234
has_test_evidence: 5000 / 5000
has_source_patch: 5000 / 5000
```

使用 Qwen3-30B-A3B tokenizer 统计的 corpus token 分布：

```text
examples: 5000
total tokens: 11269454
avg tokens/example: 2253.9
median: 1981.5
p90: 3379
p95: 4192
p99: 6784
max: 12691
over 4096: 272 examples, 5.44%
over 8192: 27 examples
```

解释：该规模足够做 LoRA ablation，但还不足以支持 full-parameter midtrain 结论。

## 训练结果

训练完成：

```text
global_steps: 563
repair_lora val_loss: 0.7844527311325074
```

Repair-domain validation loss：

```text
base repair val_loss:        1.2445924415588379
repair_lora repair val_loss: 0.7844527311325074
delta:                       -0.4601397104263305
```

LoRA 明显降低 repair-domain validation loss，说明该 PR repair corpus 对 Qwen3-30B-A3B 有稳定的修复域训练信号。

## A/B 评测结果

评测配置：

```text
eval root: outputs/qwen3_30b_a3b_side/ab_eval_parallel
repair validation split: deterministic 10% split
HumanEval: local greedy pass@1
MBPP: local greedy pass@1 with Qwen chat template and enable_thinking=False
```

最终结果：

```text
base HumanEval:        52 / 164 = 31.71%
repair_lora HumanEval: 57 / 164 = 34.76%
delta HumanEval:       +5 tasks, +3.05 percentage points

base MBPP:             405 / 974 = 41.58%
repair_lora MBPP:      500 / 974 = 51.33%
delta MBPP:            +95 tasks, +9.75 percentage points
```

Summary JSON:

```text
outputs/qwen3_30b_a3b_side/ab_eval_parallel/ab_summary.json
```

## 和 Qwen3-8B 5k 结果对比

同一 5k high-quality PR repair corpus 上，Qwen3-8B 与 Qwen3-30B-A3B 都显示 repair loss 下降，但 30B-A3B 的 HumanEval / MBPP 改善更干净：

```text
Qwen3-8B 5k:
repair loss: 1.4626 -> 0.7734
HumanEval:   10.37% -> 15.24%
MBPP chat:   67.97% -> 67.15%

Qwen3-30B-A3B 5k:
repair loss: 1.2446 -> 0.7845
HumanEval:   31.71% -> 34.76%
MBPP chat:   41.58% -> 51.33%
```

注意：两个 base model 的 tokenizer、chat behavior 和预训练分布不同，不能只看绝对 MBPP 数值横向比较。更可靠的是同一 base 内部的 before/after delta。

## 当前判断

Qwen3-30B-A3B 的 5k LoRA ablation 是当前最干净的正向结果：

- repair validation loss 从 `1.2446` 降到 `0.7845`；
- HumanEval 从 `31.71%` 提升到 `34.76%`；
- MBPP 从 `41.58%` 提升到 `51.33%`；
- smoke test 输出是正常 Python，没有 TeleBase3 分支出现的 repeated-token / tokenizer failure。

这支持继续做 scaling validation，但不建议直接跳到 50k full midtrain。

## 建议下一步

下一步先扩展到 10k high-quality corpus，并复用同一 Qwen3-30B-A3B LoRA 配方：

```text
target corpus: 10000 high-quality PR repair records
model: /gfs/space/private/xb/model/Qwen3-30B-A3B-Instruct-2507
method: LoRA
epochs: 1
lr: 5e-5
max_length: 4096
batch_size: 1
grad_accum: 8
eval: repair val loss + HumanEval + MBPP
```

如果 10k 的 quality、token stats 和 A/B 指标稳定，再继续尝试 15k 或 20k。50k 需要先扩展 repo pool 并确认 GitHub API yield，避免在数据质量和 rate limit 上盲目消耗 GPU。
