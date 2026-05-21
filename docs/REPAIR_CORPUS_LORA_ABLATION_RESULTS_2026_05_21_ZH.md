# PR Repair Corpus LoRA Ablation Results

更新时间：2026-05-21

本文汇总当前 PR repair corpus LoRA ablation 的主要结果，便于后续分析不同模型和不同数据规模的收益。所有实验均只训练 LoRA，核心配方保持一致：1 epoch，学习率 `5e-5`，最大长度 `4096`。`10k-target` 实际得到 `8850` 条高质量 PR repair 样本，因为当前 repo 池在质量过滤下已经耗尽。

## 数据规模

| 数据集 | 实际样本数 | 说明 |
|---|---:|---|
| 5k HQ | 5000 | 高质量 PR repair corpus |
| 10k-target HQ | 8850 | 目标 10000，实际 8850；所有样本有 source patch 和 test evidence |

10k-target corpus 摘要：

```text
target: 10000
actual corpus_records: 8850
sft_records: 35400
quality_avg: 0.950658418079096
has_test_evidence: 8850 / 8850
has_source_patch: 8850 / 8850
```

## 评测口径

- Repair loss：在 held-out repair corpus 上计算 validation loss，越低越好。
- HumanEval：pass@1，`164` 题。
- MBPP：pass@1，`974` 题。
- IQuest 模型的 HumanEval 使用 chat prompt + code extraction；raw HumanEval 对 IQuest 不可靠，因为模型容易输出解释文本。
- MBPP 使用 chat prompt。

## Qwen3-8B

模型路径：

```text
/gfs/space/private/wujn/Learn/models/Qwen3-8B
```

| 数据 | Repair loss Base -> LoRA | HumanEval Base -> LoRA | MBPP Base -> LoRA |
|---|---:|---:|---:|
| 1k HQ | 1.4605 -> 0.8372 | 17/164 -> 19/164 | 662/974 -> 665/974 |
| 2k | 1.4787 -> 0.8043 | 17/164 -> 22/164 | 662/974 -> 660/974 |
| 5k HQ | 1.4626 -> 0.7734 | 17/164 -> 25/164 | 662/974 -> 654/974 |
| 10k-target, actual 8850 | 1.4707 -> 0.7656 | 17/164 -> 27/164 | 663/974 -> 649/974 |

增量：

| 数据 | Repair loss delta | HumanEval delta | MBPP delta |
|---|---:|---:|---:|
| 1k HQ | -0.6233 | +2 | +3 |
| 2k | -0.6744 | +5 | -2 |
| 5k HQ | -0.6892 | +8 | -8 |
| 10k-target, actual 8850 | -0.7051 | +10 | -14 |

结论：

- Qwen3-8B 的 repair loss 随数据规模扩大稳定下降，8850 条达到当前最低 `0.7656`。
- HumanEval 也随数据规模扩大单调提升：`19 -> 22 -> 25 -> 27`。
- MBPP 出现明显 specialization trade-off：5k 已经低于 base，8850 进一步回落到 `649/974`。
- 这组补齐后可以更清楚地看到：小模型从 repair corpus 获得更强修复能力和 HumanEval 收益，但对通用 MBPP 有负迁移。

输出路径：

```text
outputs/qwen3_8b_repair_1k_hq_ab_eval_parallel/ab_summary.json
outputs/qwen3_8b_repair_2k_ab_eval_parallel/ab_summary.json
outputs/qwen3_8b_repair_5k_hq_ab_eval_parallel/ab_summary.json
outputs/qwen3_8b_repair_10k_target_hq_ab_eval_parallel/ab_summary.json
```

## Qwen3-30B-A3B-Instruct-2507

模型路径：

```text
/gfs/space/private/xb/model/Qwen3-30B-A3B-Instruct-2507
```

| 数据 | Repair loss Base -> LoRA | HumanEval Base -> LoRA | MBPP Base -> LoRA |
|---|---:|---:|---:|
| 5k HQ | 1.2446 -> 0.7845 | 52/164 -> 57/164 | 405/974 -> 500/974 |
| 10k-target, actual 8850 | 1.2508 -> 0.7797 | 52/164 -> 57/164 | 405/974 -> 504/974 |

增量：

| 数据 | Repair loss delta | HumanEval delta | MBPP delta |
|---|---:|---:|---:|
| 5k HQ | -0.4601 | +5 | +95 |
| 10k-target, actual 8850 | -0.4711 | +5 | +99 |

结论：

- Qwen3-30B-A3B 在 5k 时已经获得主要收益。
- 扩到 8850 条后，repair loss 略降，MBPP 从 `500` 提到 `504`，HumanEval 持平。
- 当前数据扩展对 MBPP 有小幅继续收益，但没有带来 HumanEval 的二次提升。

输出路径：

```text
outputs/qwen3_30b_a3b_side/ab_eval_parallel/ab_summary.json
outputs/qwen3_30b_a3b_repair_10k_hq_ab_eval_parallel/ab_summary.json
```

## IQuest-Coder-V1-14B-Instruct

模型路径：

```text
/gfs/space/private/wujn/models/IQuest-Coder-V1-14B-Instruct
```

| 数据 | Repair loss Base -> LoRA | HumanEval Base -> LoRA | MBPP Base -> LoRA |
|---|---:|---:|---:|
| 5k HQ | 1.0036 -> 0.5844 | 101/164 -> 109/164 | 607/974 -> 623/974 |
| 10k-target, actual 8850 | 1.0062 -> 0.5737 | 101/164 -> 109/164 | 607/974 -> 606/974 |

增量：

| 数据 | Repair loss delta | HumanEval delta | MBPP delta |
|---|---:|---:|---:|
| 5k HQ | -0.4191 | +8 | +16 |
| 10k-target, actual 8850 | -0.4325 | +8 | -1 |

结论：

- IQuest 14B 的 5k ablation 成功：repair loss 明显下降，HumanEval 和 MBPP 都提升。
- 10k-target 继续降低 repair loss，并保持 HumanEval 的 `+8` 题提升。
- 但 10k-target 的 MBPP 没有继续提升，反而从 5k LoRA 的 `623/974` 回落到 `606/974`，接近 base。
- 当前信号更像是 repair objective 学得更强，但对通用 MBPP 的收益在 14B 上没有随数据规模单调增加。

输出路径：

```text
outputs/iquest_coder_v1_14b_5k_side/ab_eval_parallel_chat_v1_fsdp4_len4096_retry2/ab_summary.json
outputs/iquest_coder_v1_14b_10k_target_side/ab_eval_parallel_chat_v1_fsdp4_len4096/ab_summary.json
```

## IQuest-Coder-V1-7B-Instruct

模型路径：

```text
/gfs/space/private/wujn/models/IQuest-Coder-V1-7B-Instruct
```

| 数据 | Repair loss Base -> LoRA | HumanEval Base -> LoRA | MBPP Base -> LoRA |
|---|---:|---:|---:|
| 5k HQ | 1.1541 -> 0.6547 | 87/164 -> 79/164 | 395/974 -> 486/974 |
| 10k-target, actual 8850 | 1.1587 -> 0.6444 | 87/164 -> 83/164 | 395/974 -> 514/974 |

增量：

| 数据 | Repair loss delta | HumanEval delta | MBPP delta |
|---|---:|---:|---:|
| 5k HQ | -0.4994 | -8 | +91 |
| 10k-target, actual 8850 | -0.5143 | -4 | +119 |

结论：

- IQuest 7B 的 repair loss 和 MBPP 对 PR repair corpus 很敏感，10k-target 比 5k 继续提升 MBPP：`486/974 -> 514/974`。
- HumanEval chat 仍低于 base，但 10k-target 的下降幅度小于 5k：`-8` 题改善到 `-4` 题。
- 7B 的现象与 14B 不同：7B 在 MBPP 上随数据规模扩大继续收益，但 HumanEval 仍有 trade-off。

输出路径：

```text
outputs/iquest_coder_v1_7b_5k_side/humaneval_chat_token_slice_v1/ab_summary.json
outputs/iquest_coder_v1_7b_5k_side/ab_eval_parallel_token_slice_v2/ab_summary.json
outputs/iquest_coder_v1_7b_10k_target_side/ab_eval_parallel_chat_v1/ab_summary.json
```

## 横向结论

| 模型 | 数据规模 | Repair loss | HumanEval | MBPP | 简要判断 |
|---|---:|---:|---:|---:|---|
| Qwen3-8B | 1k | 大幅下降 | +2 | +3 | 小幅正向 |
| Qwen3-8B | 2k | 大幅下降 | +5 | -2 | HE 涨，MBPP 微降 |
| Qwen3-8B | 5k | 大幅下降 | +8 | -8 | HE 涨，MBPP 回落 |
| Qwen3-8B | 8850 | 大幅下降 | +10 | -14 | HE 继续涨，MBPP 继续回落 |
| Qwen3-30B-A3B | 5k | 大幅下降 | +5 | +95 | 主要收益已出现 |
| Qwen3-30B-A3B | 8850 | 略优于 5k | +5 | +99 | MBPP 小幅续涨，HumanEval 持平 |
| IQuest 14B | 5k | 大幅下降 | +8 | +16 | 三项均正向 |
| IQuest 14B | 8850 | 略优于 5k | +8 | -1 | repair 更强，MBPP 回落 |
| IQuest 7B | 5k | 大幅下降 | -8 | +91 | MBPP 强收益，HumanEval 下降 |
| IQuest 7B | 8850 | 略优于 5k | -4 | +119 | MBPP 继续涨，HumanEval trade-off 缓和 |

总体判断：

- PR repair corpus 对 repair validation loss 的改善在所有模型上都稳定。
- MBPP 收益最稳定的是 Qwen3-30B-A3B 和 IQuest 7B；Qwen3-8B 与 IQuest 14B 在 10k-target 上都出现 MBPP trade-off。
- HumanEval 对模型和评测口径更敏感。Qwen3-30B-A3B 与 IQuest 14B 保持正向，IQuest 7B 仍有下降。
- 下一步如果继续扩数据，建议先扩 repo 池并维持质量过滤，再做一个 `15k/20k` 点；不要直接假设 `50k` 会在所有指标上单调提升。
