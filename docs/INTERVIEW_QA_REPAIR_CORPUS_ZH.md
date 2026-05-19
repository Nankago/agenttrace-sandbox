# 代码修复中训练数据项目面试问答

这份文档用于面试时解释 `agenttrace-sandbox` 中 GitHub PR-derived repair corpus 的数据建设、训练验证和评测排查。回答要点是：不夸大成完整大规模训练系统，强调数据管线、质量控制、训练原型和评测闭环。

## 1. 这个项目一句话怎么介绍？

我做的是一个面向代码修复中训练数据的数据构造与验证项目：从 GitHub Bug Fix PR、关联 Issue 和 diff 中抽取 evidence-grounded repair cards，再导出 midtrain corpus / SFT 数据，并用 Qwen3-8B + LoRA 做 repair-domain continued-pretraining 原型验证。

更简洁的说法：

```text
构建 GitHub PR/Issue Bug Fix 数据 -> Repair Cards -> midtrain corpus / SFT 的端到端链路，并验证这类 PR-derived repair corpus 对代码修复领域适配是否有训练信号。
```

## 2. 这是不是 SWE Agent 训练数据？

严格说不是主线。这个项目原仓库支持 agent trace / tool-call SFT，但这次主要工作不是收集 SWE Agent trajectory，而是构建 GitHub PR-derived repair corpus。

面试里建议这样区分：

```text
项目里有 agent trace 能力，但这次实验主线是 PR-derived code repair corpus，偏中训练/continued pretraining 数据，而不是 agent trajectory 数据。
```

## 3. 为什么从 GitHub PR/Issue 构造数据？

Bug Fix PR 天然包含代码修复任务需要的几个关键部分：

- Issue / PR 描述：bug symptom、触发条件、用户反馈。
- source diff：实际修复发生在哪些文件、改了什么逻辑。
- test diff：新增或修改的测试，相当于 expected behavior / oracle。
- repo metadata：项目、PR 号、文件路径、质量信号。

这些信息比单纯代码片段更接近真实软件修复场景，适合构造成 repair reasoning / repair-domain corpus。

## 4. 当前 PR 是怎么抓取的？

脚本是 `scripts/build_pr_repair_experiment.py`。

抓取流程：

```text
1. 预设一批高质量 Python / ML / Web 框架仓库
2. 对每个 repo 拉 closed PR 列表，默认最多看 500 个 PR
3. 用 PR title/body 做 bug-fix 关键词预过滤
4. 对候选 PR 拉 unified diff
5. 如果 PR 明确 Fixes/Closes/Resolves 某个 issue，再拉 issue
6. 基于 PR + issue + diff 做 bug-fix 质量打分
7. 转成 Repair Card
8. 导出 midtrain corpus 和 SFT 数据
```

GitHub API 主要调用：

```text
GET /repos/{repo}/pulls?state=closed&per_page=100&sort=updated&direction=desc&page=...
GET /repos/{repo}/pulls/{number} with Accept: application/vnd.github.v3.diff
GET /repos/{repo}/issues/{issue_number}
```

## 5. PR 和 Issue 是如何关联的？

只识别明确 closing reference：

```text
Fixes #123
Closes #123
Resolves #123
Closes https://github.com/owner/repo/issues/123
```

不会把普通 `#123` 当作 linked issue，也会忽略 HTML comment 和 code block 里的模板示例。例如：

```text
<!-- Example: Fixes #999 -->
```

这个优化很关键，因为 PR 模板里经常有示例 `Fixes #...`，如果误抓，会把错误 issue 和正确 diff 拼成一条伪 repair 样本，后续 LLM enrich 很容易生成高置信但错误的 root cause。

## 6. bug-fix 过滤规则是什么？

核心过滤包括：

```text
bug_fix_score >= 3.0
is_bug_fix = true
必须有 source_files
必须有 test_files
不能 docs-only
不能 tests-only
不能 dependency-only
不能 low-signal-only
不能 large patch
repair card quality >= 0.85
```

打分会看：

- bug/fix/regression/error/exception/failing 等正向关键词；
- 是否有 linked issue；
- 是否修改 source file；
- 是否同时修改 test file；
- 是否包含 docs/refactor/ci/dependency 等负向信号；
- patch 是否过大或低信号。

## 7. 什么是 Repair Card？

Repair Card 是把原始 PR/Issue/diff 结构化后的中间表示。典型字段包括：

```text
symptom
localization
patch_intent
test_oracle
validation
quality
evidence
```

其中 `evidence` 会保存 issue text、PR text、source diff、test diff 等，每个 evidence 有 `E1/E2/...` ID。后续 SFT 或 midtrain 文本都从这个结构导出。

## 8. SFT 数据和 midtrain 数据有什么区别？

SFT 数据是 instruction-output 格式，比如：

```text
Instruction: Identify source/test files involved in fixing the bug.
Input: evidence packet
Output: source_files / test_files
```

midtrain corpus 是连续文本格式，把 repo、PR、evidence、symptom、root cause、patch intent、test oracle 等组织成一段可做 causal LM 的文本。

这次训练主线用的是 midtrain corpus，也就是 continued pretraining / DAPT 风格，不是 instruction SFT。

## 9. 为什么说这是中训练 / continued pretraining 原型？

训练方式是 causal language modeling：

```text
input_ids = tokenizer(text + eos)
labels = input_ids
loss = model(**batch).loss
```

它训练模型预测 repair corpus 的下一个 token，不是让模型直接回答某个 instruction。因此更准确叫：

```text
repair-domain continued pretraining / DAPT prototype
```

如果简历里不想写“后训练”，可以写“代码修复中训练数据建设与原型验证”。

## 10. repair loss 是怎么计算的？

`repair loss` 是在 repair corpus 的 held-out validation split 上算 causal LM loss。

流程：

```text
1. 读取 corpus JSONL 中每行 row["text"]
2. seed=7 随机打乱
3. 取 10% 做 validation
4. 对 validation text 做 next-token prediction
5. 对所有 validation examples 的 loss 取平均
```

它衡量模型是否更会预测 GitHub repair corpus 的文本分布，不直接等价于“能不能修 bug”。

## 11. 第一阶段 500 PR 实验结果是什么？

第一阶段数据：

```text
first-stage selected PR: 500
repair cards strict: 476
midtrain corpus records: 465
```

训练设置：

```text
Model: Qwen3-8B
Adapter: LoRA
epochs: 3
lr: 1e-4
GPU: H800
```

结果：

```text
base repair val_loss:        1.5629
repair_lora repair val_loss: 0.7829
```

解释：

```text
LoRA 明显学到了 repair corpus 分布，但这只是 domain adaptation 指标，需要结合 MBPP/HumanEval 等评测看通用代码能力是否保持。
```

## 12. MBPP / HumanEval 结果怎么看？

一开始 raw-prompt MBPP 结果异常差：

```text
base MBPP:        4.72%
repair_lora MBPP: 0.82%
```

后来排查发现是 Qwen3 prompt format 问题：Qwen3 应使用 chat template，raw prompt 会导致模型输出解释、重复 prompt 或非代码文本。

修正后 MBPP chat-template 结果：

```text
base MBPP:        662 / 974 = 67.97%
repair_lora MBPP: 657 / 974 = 67.45%
```

结论：

```text
repair loss 明显下降，MBPP chat-template 下通用代码能力基本保持，仅轻微下降约 0.52 个百分点。
```

HumanEval 当前是本地简化 prefix-completion smoke test，不建议当主流 benchmark 结论引用。

## 13. Qwen3-8B 的 MBPP 结果符合主流评测吗？

修正后的 MBPP 大体符合。公开 Qwen3-8B MBPP 量级约 69.8，我们本地 chat-template MBPP 是 67.97%，差异可能来自 harness、prompt、decoding、数据版本和执行环境。

面试里可以说：

```text
本地 MBPP chat-template 结果与公开 Qwen3-8B MBPP 指标量级一致；HumanEval 当前只是 smoke test，不作为正式 benchmark 结论。
```

## 14. 为什么第二阶段目标 2k，最后只有 1470 条？

第二阶段目标是抓 2k，但新的高精度过滤更严格：

```text
quality >= 0.85
必须有 source patch
必须有 test evidence
PR-issue 错配过滤
过滤 docs/test-only/dependency/large/low-signal
```

最终 30 个 repo 得到：

```text
selected_prs:   1702
repair_cards:   1470
corpus_records: 1470
sft_records:    5880
quality_avg:    0.9508
```

这说明数据精度提升，但召回下降。对于中训练，少量高质量样本通常比混入大量错配样本更安全。

## 15. 为什么会遇到 GitHub rate limit？

GitHub authenticated core API 通常是 5000 requests/hour。抓 PR 不是每个 PR 一次请求：

```text
PR list pages
PR diff
linked issue
```

一个候选 PR 可能消耗 1-2 次请求，多个 repo、每个 repo 500 PR 很容易打满。

处理方式：

```text
1. repo 级 JSONL cache
2. 遇到 rate limit 自动等待 reset
3. 重启时复用 cache
```

## 16. 现在有没有用 DeepSeek / LLM enrich？

当前 500 和第二阶段 1470 都是 rule-only repair cards，没有走 DeepSeek enrich。

原因是先做可控 baseline：

```text
rule-only corpus -> train -> eval
```

后续可以做 ablation：

```text
500 rule-only vs 500 DeepSeek-enriched
```

再判断 enrich 是否真的提升 repair loss 或下游 repair behavior。

## 17. enrich 可能提升质量吗？

有机会提升，主要提升 semantic repair fields：

```text
root_cause
failure_condition
expected_behavior
repair_rationale
edge_cases
```

但有风险：

```text
1. PR/issue 错配时会生成错误解释
2. LLM 可能 hallucinate root cause
3. 过多 prose 可能污染代码生成风格
4. 成本和吞吐上升
```

所以必须做 grounding：

```text
每个字段必须引用 evidence_ids
无证据写 insufficient_evidence
保留原始 source/test diff
只把 LLM enrich 作为补充，不替代 evidence
```

## 18. 你做过哪些关键质量优化？

可以总结为：

```text
1. bug-fix 过滤：过滤 docs-only / tests-only / dependency-only / large patch / low-signal patch
2. diff 清洗：去掉 lock/vendor/generated/binary/过长低信号 diff
3. PR 模板清洗：降低 boilerplate 对 repair text 的污染
4. PR-issue 错配过滤：只识别真实 closing reference，忽略模板示例
5. issue-pr consistency score：弱相关 issue 降权
6. Qwen3 eval harness 修正：MBPP 改用 chat template
```

## 19. 这个项目的主要局限是什么？

可以坦诚说：

```text
1. 当前主要是 rule-only repair cards，semantic fields 还不如 LLM-enriched 精细
2. repair loss 是 domain adaptation 指标，不等价于真实 patch 修复成功率
3. HumanEval 当前是简化 smoke test，不作为正式 benchmark
4. 2k 目标在严格过滤后实际得到 1470 条，需要更多 repo 或放宽策略
5. 还没有跑 SWE-bench Lite 级别的真实 issue-to-patch benchmark
```

这不是缺点，而是后续路线。

## 20. 下一步怎么做？

优先级建议：

```text
1. 固定 1470 high-precision corpus，完成 1 epoch / lr=5e-5 LoRA A/B eval
2. 做 500 rule-only vs 500 DeepSeek-enriched ablation
3. 增加更多 repo，扩展到 3k-10k 高质量 repair cards
4. 增加 patch-generation / SWE-bench Lite 小规模评测
5. 比较 prose-heavy、diff-heavy、test-heavy 不同 corpus 格式
```

## 21. 简历上最稳的一句话

```text
构建 GitHub PR/Issue Bug Fix 数据到 Repair Cards、midtrain corpus 与 SFT 数据的端到端链路，覆盖 PR 抓取、Issue 关联、diff 解析、bug-fix 过滤、模板噪声清洗和 PR-issue 错配过滤；基于 Qwen3-8B + LoRA 在 4x H800 上完成 repair-domain 中训练原型验证，repair held-out loss 从 1.56 降至 0.78，并通过 MBPP chat-template A/B 评测分析通用代码能力保持情况。
```

如果想更保守：

```text
构建 GitHub PR-derived 代码修复语料管线，并完成 Qwen3-8B LoRA repair-domain 训练验证与评测排查。
```
