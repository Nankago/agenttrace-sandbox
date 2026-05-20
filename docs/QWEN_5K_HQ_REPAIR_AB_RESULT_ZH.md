# Qwen3-8B 5k High-Quality PR Repair Corpus A/B Result

## 实验目标

本实验用于验证：当 GitHub PR-derived repair corpus 从 1k/1470 扩展到 5k 后，Qwen3-8B 的代码修复训练信号是否继续增强，以及是否会伤害通用代码生成能力。

训练配置保持不变：

```text
base model: /gfs/space/private/wujn/Learn/models/Qwen3-8B
method: LoRA
epochs: 1
lr: 5e-5
max_length: 4096
batch_size: 1
grad_accum: 8
train corpus: data/experiments/second_stage_5k_hq/corpus/repair_corpus_semantic.jsonl
adapter: outputs/qwen3_8b_repair_corpus_lora_5k_hq_e1_lr5e5/adapter
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

仓库来源覆盖 70+ 个 Python 生态项目。高频来源包括：

```text
pandas-dev/pandas:       215
sympy/sympy:             193
pallets/click:           184
pydantic/pydantic:       159
ansible/ansible:         156
tox-dev/tox:             155
scikit-learn/scikit-learn: 148
modin-project/modin:     144
sqlalchemy/sqlalchemy:   142
twisted/twisted:         140
```

## 训练结果

训练完成：

```text
global_steps: 563
repair_lora val_loss: 0.7734282069206238
```

Repair-domain validation loss：

```text
base repair val_loss:        1.462557979106903
repair_lora repair val_loss: 0.7734282069206238
delta:                       -0.6891297721862792
```

这是当前规模曲线里最强的 repair loss 改善。

## A/B 评测结果

评测配置：

```text
eval root: outputs/qwen3_8b_repair_5k_hq_ab_eval_parallel
repair validation split: deterministic 10% split
HumanEval: local greedy pass@1
MBPP: local greedy pass@1 with Qwen chat template and enable_thinking=False
```

最终结果：

```text
base HumanEval:        17 / 164 = 10.37%
repair_lora HumanEval: 25 / 164 = 15.24%
delta HumanEval:       +4.88 percentage points

base MBPP:             662 / 974 = 67.97%
repair_lora MBPP:      654 / 974 = 67.15%
delta MBPP:            -0.82 percentage points
```

Summary JSON:

```text
outputs/qwen3_8b_repair_5k_hq_ab_eval_parallel/ab_summary.json
```

## 规模曲线

当前 scale curve：

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

5000 high-quality:
repair loss: 1.4626 -> 0.7734
HumanEval:   10.37% -> 15.24%
MBPP chat:   67.97% -> 67.15%
```

结论：

- 5k 继续降低 repair loss，说明 PR-derived repair corpus 的修复域训练信号随规模增强。
- HumanEval 提升最大，说明修复语料没有破坏基础代码能力，并可能增强了部分 Python 函数级推理。
- MBPP 出现 `-0.82` percentage points 的小幅退化，提示模型开始向 repair/diff/workflow 风格轻微专门化。

## 为什么 PR Repair 数据和 MBPP 不完全契合

这个 MBPP 结果不应简单解读为“数据质量差”。更合理的解释是：PR repair corpus 和 MBPP 的任务分布不同。

### 1. 任务形式不同

MBPP 是短题目、单函数、从零生成代码：

```text
input: natural language problem + tests
output: standalone Python function/program
```

PR repair corpus 是真实仓库修复：

```text
input: issue / PR text / evidence / diff context / changed source and test files
output: repair rationale or patch-oriented response
```

训练语料中每条记录平均包含多个真实文件、测试证据和 diff 语境。它更像“理解 bug、定位文件、解释或生成修复”，不是 MBPP 的“短算法题生成”。

### 2. 文件与项目上下文更复杂

5k repair cards 的文件统计：

```text
source files total: 10247
test files total:   7798
avg source files/card: 2.05
avg test files/card:   1.56
```

这说明样本往往涉及多个文件和真实项目上下文，而 MBPP 通常只需要一个简短函数。模型学到的上下文组织方式和 MBPP 的简短输出格式不完全一致。

### 3. 语言和文件类型不完全等同于 MBPP

虽然主体是 Python，但 repair corpus 来自真实仓库，包含配置、类型文件、前端和少量非 Python 文件：

```text
source .py:   7751
source .ts:   493
source .yml:  179
source .json: 179
source .tsx:  120
source .c:     80
source .pyi:   46
```

MBPP 是纯 Python 小题。真实 PR 数据更接近工程代码修复，分布更宽。

### 4. 输出风格可能轻微偏向 repair card

训练文本包含固定结构：

```text
Problem Summary
Evidence
Changed source files
Changed test files
Symptom
Root Cause
Patch Intent / Repair Rationale
Test Oracle
```

这些结构有利于修复任务，但可能让模型在 MBPP 这种“只输出代码”的任务上略微偏离最优格式。当前 MBPP 只下降 `0.82` percentage points，属于轻微退化，但已经能看到 specialization 信号。

## 当前判断

5k 结果整体是正向的：

- repair loss 更低；
- HumanEval 明显提升；
- MBPP 小幅下降但没有崩。

这说明当前数据适合作为代码修复 / bug-fix midtrain 数据，但不能直接等价为通用 Python 竞赛题或短函数生成数据。MBPP 更适合作为 regression check，而不是这个项目的主指标。

## 建议下一步

下一步不建议直接全量微调。更稳妥的 ablation 是：

```text
5k corpus, same LoRA setup, lower training strength:
- lr: 2e-5
- epochs: 1
- max_length: 4096
```

目标是验证：

```text
repair loss 仍明显下降；
HumanEval 保持提升；
MBPP 退化收窄或恢复。
```

如果低学习率能保住 repair/HumanEval，同时改善 MBPP，说明问题主要是训练强度而不是数据本身。如果低学习率仍伤 MBPP，则下一步再考虑混入通用 Python instruction/code replay。

