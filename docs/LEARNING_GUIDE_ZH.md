# AgentTrace Sandbox 学习指南

这份文档用通俗方式解释 AgentTrace Sandbox 是什么、怎么跑、代码如何组织，以及它为什么可以作为一个 Coding Agent 后训练项目来讲。

如果你只想先抓住一句话：

> AgentTrace Sandbox 是一个安全采集 Coding Agent 执行轨迹的实验平台：让模型在隔离工作区里读代码、改代码、跑测试，并把每一步变成可分析、可训练的数据。

## 1. 这个项目在解决什么问题

普通 Coding Agent 项目通常只关心最后答案：

```text
用户给任务 -> 模型改代码 -> 测试通过/失败
```

但如果你想做后训练，比如 SFT、DPO、RL，你更关心中间过程：

```text
模型看了哪些文件？
它为什么选这个工具？
它有没有按 JSON 协议输出？
它有没有乱改测试？
测试失败时失败在哪？
成功轨迹能不能转成训练数据？
```

AgentTrace Sandbox 的核心就是把这些过程系统性记录下来。

## 2. 项目主流程

一次任务运行大概是这样：

```text
输入任务
  -> 复制目标 repo 到 runs/<run_id>/workspace
  -> 调用模型生成计划
  -> 模型每次输出一个 JSON 工具调用
  -> 本地执行工具
  -> 记录工具结果
  -> 如果改了代码，就运行测试
  -> 记录最终 diff、outcome、trace
  -> 从 trace 导出 SFT 数据
```

可以理解成两层：

```text
第一层：Agent 执行层
让模型真的操作一个代码仓库。

第二层：数据采集层
把模型的每一步行为记录成 trace，再转成训练样本。
```

## 3. 现在已经支持什么

| 能力 | 说明 |
|---|---|
| 单任务运行 | 对一个 repo 执行一个自然语言 coding task |
| 批量任务 | 用 JSONL manifest 批量跑任务 |
| Mock 模型 | 不需要 API key，也能跑通 demo |
| 真实 API | 支持 OpenAI-compatible chat completions，例如 DeepSeek |
| 工具调用 | 支持读文件、搜索、替换、写文件、跑测试、看 diff |
| 安全限制 | 路径不能逃出 workspace，危险命令会被拦截 |
| Docker 后端 | 测试命令可在 Docker 容器中执行 |
| Trace 记录 | 保存 raw output、工具调用、测试结果、耗时、最终 diff |
| 失败分类 | 记录 success、invalid_json、edit_miss、test_failed 等 outcome |
| 格式检测 | 标记模型是否输出了非纯 JSON 的可恢复格式 |
| SFT 导出 | 支持 JSONL 和 Alpaca 格式 |
| Stats | 统计通过率、失败分布、平均步数、格式违规率 |
| Benchmark 构建 | 把带 unit test 的函数任务生成可运行 repo 和 manifest |
| MBPP/HumanEval 构建 | 把公开代码 benchmark 转成可执行的函数补全任务 |
| Unit Test 补全构建 | 从已有 Python repo 的测试 import 中挖空函数，生成补全任务 |
| GitHub 自动抓取 | 从 GitHub repo 抓取 PR、关联 issue 和 diff |
| PR/Issue Wiki | 把 PR/Issue/diff JSONL 转成结构化修复解释数据 |
| Repair Card 统计 | 统计质量分、测试证据、源码 patch 和 LLM grounding |
| Repair Corpus 导出 | 把 Repair Card 线性化为中训练 / continued pretraining 文本 |
| Repair SFT 变体 | 支持 full、no-llm、no-tests、diff-only ablation |

## 4. 快速跑通

进入项目：

```bash
cd agenttrace-sandbox
```

跑 mock demo：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli run \
  --repo examples/buggy_calculator \
  --task "Fix the subtract function bug." \
  --test-command "python3 -m unittest discover -s tests" \
  --mock
```

你会看到类似：

```text
outcome=success
steps=5
trace=runs/<run_id>/trace.jsonl
summary=Fixed subtract function bug...
```

批量跑 manifest：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli run-manifest \
  --manifest examples/tasks.jsonl \
  --output runs/manifest_results.jsonl \
  --mock
```

导出 SFT 数据：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli export-sft \
  --traces runs \
  --output data/sft/tool_calls.jsonl
```

导出 Alpaca 格式：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli export-sft \
  --traces runs \
  --output data/sft/tool_calls_alpaca.json \
  --format alpaca
```

查看统计：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli stats --runs runs
```

构造 benchmark 风格任务：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli build-benchmark \
  --output-dir data/benchmarks/offline \
  --limit 5
```

构造 MBPP 任务：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli build-mbpp \
  --output-dir data/benchmarks/mbpp \
  --limit 20 \
  --dataset-source auto
```

构造 HumanEval 任务：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli build-humaneval \
  --output-dir data/benchmarks/humaneval \
  --limit 20 \
  --dataset-source auto
```

`--dataset-source` 可以选：

```text
auto        优先尝试 ModelScope，再尝试 Hugging Face，最后 fallback 到内置样例
modelscope  只尝试 ModelScope，失败则 fallback 到内置样例
huggingface 只尝试 Hugging Face，失败则 fallback 到内置样例
offline     只使用内置样例
```

如果你想从 Hugging Face 拉取公开数据集，可以安装：

```bash
pip install -e ".[benchmarks]"
```

如果你想从 ModelScope 拉取数据集，可以安装：

```bash
pip install -e ".[modelscope]"
export MODELSCOPE_SDK_TOKEN="your_token_here"
```

也支持用 `MODELSCOPE_API_TOKEN` 这个环境变量名。

默认 ModelScope 数据集 id 是：

```text
MBPP: OmniData/MBPP
HumanEval: openai-mirror/openai_humaneval
```

如果你要换成别的数据集，可以加 `--modelscope-dataset <dataset_id>`。

这会生成：

```text
data/benchmarks/<name>/tasks.jsonl
data/benchmarks/<name>/repos/<task_id>/solution.py
data/benchmarks/<name>/repos/<task_id>/tests/test_solution.py
```

其中 `tasks.jsonl` 是 agent 要处理的任务列表，`repos/<task_id>` 是每个任务对应的独立小代码仓库。

从已有 Python repo 构造 Unit Test 函数补全任务：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli build-unit-completion \
  --repo examples/buggy_calculator \
  --output-dir data/benchmarks/unit_completion \
  --limit 20 \
  --min-confidence 0.5 \
  --max-per-file 5 \
  --check-baseline
```

它会扫描 `tests/test*.py`，先用 AST 确认 import 进来的函数确实被测试调用，再复制 repo、挖空目标函数体并生成 manifest。现在支持这些常见写法：

```python
from calculator import subtract
subtract(3, 1)

import calculator
calculator.subtract(3, 1)

import calculator as calc
calc.subtract(3, 1)

from calculator import subtract as minus
minus(3, 1)

from pkg import math_utils
math_utils.add(1, 2)
```

模块定位支持普通文件、`src` layout 和 package layout，例如：

```text
math_utils.py
src/pkg/math_utils.py
pkg/math_utils.py
pkg/__init__.py
```

挖空时会保留函数签名、decorator 和 docstring。默认生成 top-level function 任务；如果测试里能直接识别 class/static method，可以用 `--include-methods` 打开方法任务。语法解析失败、目标文件不存在、挖空后不是合法 Python 的目标会被跳过，不会让整个 builder 崩掉。

生成任务前，builder 默认会先跑相关 baseline 测试，例如：

```bash
python3 -m unittest tests.test_calculator.CalculatorTests.test_add
```

如果原始 repo 里这个相关测试本来就失败，就跳过该目标，避免“函数补对了但被无关旧 bug 拖失败”的脏样本。对于 `src/` layout，本地测试执行会自动把 `src/` 加进 `PYTHONPATH`。

常用参数：

| 参数 | 说明 |
|---|---|
| `--min-confidence` | 只保留置信度不低于阈值的目标，默认 `0.5` |
| `--include-methods` | 允许生成 class/static method 任务 |
| `--exclude-private` / `--no-exclude-private` | 默认跳过 `_private` 函数 |
| `--max-per-file` | 每个源码文件最多生成多少个任务，默认 `5` |
| `--check-baseline` / `--no-check-baseline` | 默认要求相关 baseline 测试先通过 |
| `--baseline-timeout` | 每个 baseline 检查的超时时间，默认 `30` 秒 |

manifest 每行会包含：

```text
source, target_file, target_symbol, test_files, confidence,
test_selectors, original_module, original_import, test_command,
full_test_command, baseline_checked, baseline_ok, baseline_command,
target_class(如果有)
```

`run-manifest` 的结果里还会增加简单的 `failure_attribution`，用于区分 `passed`、`baseline_failure`、`model_or_task_failure` 和 `unknown`。

这样得到的任务就是：

```text
给 agent 一个真实测试约束，让它补全函数，使原有单测通过。
```

可以先 dry-run 验证 manifest：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli run-manifest \
  --manifest data/benchmarks/unit_completion/tasks.jsonl \
  --output runs/unit_completion_results.jsonl \
  --dry-run
```

真实跑完后再导出 SFT：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli export-sft \
  --traces runs \
  --output data/sft/unit_completion_tool_calls.jsonl
```

构造 PR/Issue Wiki 数据：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli build-pr-wiki \
  --input examples/pr_issue_pairs.jsonl \
  --output data/wiki/repair_wiki.jsonl
```

从 GitHub 自动抓取 PR/Issue/diff：

```bash
export GITHUB_TOKEN="your_github_token"

PYTHONPATH=src python3 -m agenttrace_sandbox.cli fetch-github-prs \
  --repo owner/name \
  --output data/github/pr_issue_pairs.jsonl \
  --limit 20 \
  --state closed
```

公共仓库不一定需要 `GITHUB_TOKEN`，但建议设置，否则容易遇到 GitHub API rate limit。抓取后的 JSONL 可以继续喂给 `build-pr-wiki`。

抓取更高质量的 bug-fix PR：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli fetch-github-prs \
  --repo django/django \
  --output data/github/django_bugfix_prs.jsonl \
  --limit 50 \
  --state closed \
  --bug-fix-only \
  --min-bug-score 2
```

`GITHUB_TOKEN` 只从环境变量读取，不要写进代码、测试或输出文件。公共仓库可以不设置 token，但有 token 时更不容易被 rate limit。

这个过滤器会综合 PR title、PR body、关联 issue 的 title/body、changed files 和 diff 内容。它会偏向 `fix`、`bug`、`regression`、`error`、`exception`、`crash`、`failing`、`traceback`、`TypeError`、`ValueError` 等信号，同时默认过滤 docs-only、CI/config-only、typo/documentation、refactor/cleanup、test-only PR。需要放宽时，可以加：

```bash
--include-docs-only
--include-tests-only
```

每条 JSONL 会多出这些质量字段：

| 字段 | 含义 |
|---|---|
| `is_bug_fix` | 是否通过默认 bug-fix 启发式 |
| `bug_fix_score` | bug-fix 评分，可配合 `--min-bug-score` 调阈值 |
| `bug_fix_reasons` | 分数来源和过滤原因 |
| `source_files` | 被认为是源码的变更文件 |
| `test_files` | 被认为是测试的变更文件 |
| `docs_only` | 是否只改文档或 CI/config |
| `tests_only` | 是否只改测试、不改源码 |

接到 repair wiki：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli build-pr-wiki \
  --input data/github/django_bugfix_prs.jsonl \
  --output data/wiki/django_repair_wiki.jsonl
```

如果输入里有 `bug_fix_score` / `bug_fix_reasons`，`build-pr-wiki` 会把它们放进 wiki 的 metadata/source_context，后面做分析或二次筛选更方便。

生成更结构化的 Repair Card：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli build-repair-cards \
  --input data/github/django_bugfix_prs.jsonl \
  --output data/wiki/django_repair_cards.jsonl \
  --min-quality 0.6
```

Repair Card 是轻量 wiki 的结构化升级版。每条记录会包含：

| 字段 | 含义 |
|---|---|
| `evidence` | 带编号的证据片段，例如 `issue_text`、`pr_text`、`source_diff`、`test_diff` |
| `repair_card` | 有证据绑定的 `symptom`、`localization`、`patch_intent`、`test_oracle`、`validation` |
| `quality` | 自动质量分，包括 `has_test_evidence`、`evidence_coverage`、`grounding_score`、`overall` |
| `derived_tasks` | 后续可派生 localization、bug explanation、repair instruction、test spec 等 SFT 样本 |

这个 JSONL 更适合作为中间数据资产：先保留结构化证据和质量分，再按需要导出 midtrain 文本、instruction SFT 或 agent task prompt。

可选：用 OpenAI-compatible 模型增强语义字段：

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export OPENAI_MODEL="deepseek-chat"

PYTHONPATH=src python3 -m agenttrace_sandbox.cli enrich-repair-cards \
  --input data/wiki/django_repair_cards.jsonl \
  --output data/wiki/django_enriched_repair_cards.jsonl \
  --limit 20
```

增强后会新增 `llm_repair_card`，包含 `root_cause`、`failure_condition`、`expected_behavior`、`repair_rationale`、`edge_cases`。每个非空字段都必须引用已有 `evidence_ids`。同时会新增 `llm_quality`，记录 JSON 是否有效、引用的证据 ID 是否存在、grounding 校验是否通过。API key 只通过环境变量读取，不要写进代码或提交。

把增强后的 Repair Card 导出成 instruction SFT：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli export-repair-sft \
  --input data/wiki/django_enriched_repair_cards.jsonl \
  --output data/sft/repair_sft.jsonl \
  --tasks localize_files,explain_bug,repair_rationale,test_spec,repair_instruction \
  --min-quality 0.7 \
  --require-grounding
```

每条 SFT 样本包含 `instruction`、`input`、`output`，metadata 里保留 `task_type`、`source_id`、`quality` 和引用的 evidence IDs。这一步就是从结构化 Repair Card 到可训练 SFT 数据的出口。

统计 Repair Card 质量：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli stats-repair-cards \
  --input data/wiki/django_repair_cards.jsonl
```

也可以传 glob，例如：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli stats-repair-cards \
  --input "data/wiki/*_repair_cards.jsonl" \
  --json
```

输出会包含记录数、`quality.overall` 的 avg/min/max、有测试证据比例、有源码 patch 比例、平均 `bug_fix_score`、可选的 `docs_only` / `tests_only`，如果有 `llm_quality` 还会统计 `valid_json`、`grounding_ok` 和平均 `field_coverage`。

导出中训练语料 Repair Corpus：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli export-repair-corpus \
  --input data/wiki/django_enriched_repair_cards.jsonl \
  --output data/corpus/django_repair_corpus.jsonl \
  --min-quality 0.7 \
  --require-grounding
```

这个输出不是 instruction/input/output，而是连续文本 JSONL。每行包含 `id`、`repo`、`text` 和 `metadata`。`text` 会线性化 Repository、PR、Issue、Problem Summary、带 evidence id 的证据、源码文件、测试文件、Symptom、Root Cause、Failure Condition、Expected Behavior、Patch Intent / Repair Rationale、Test Oracle、Edge Cases 和质量信号。默认不会把巨大 diff 原样塞进文本，只保留摘要；可以用 `--max-evidence-chars` 控制证据长度，只有明确需要时才使用 `--include-raw-diff`。

导出 Repair SFT 时可以做 ablation：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli export-repair-sft \
  --input data/wiki/django_enriched_repair_cards.jsonl \
  --output data/sft/django_repair_sft_no_llm.jsonl \
  --variant no-llm
```

四个变体：

| variant | 含义 |
|---|---|
| `full` | 默认，优先使用 LLM enriched fields 和 evidence |
| `no-llm` | 只使用规则构建的 `repair_card` / `derived_tasks` |
| `no-tests` | 移除 test evidence，并跳过 `test_spec` |
| `diff-only` | 只用 PR 文本和源码 diff 做弱 baseline |

推荐实验设计：

| 实验 | 目的 |
|---|---|
| Base Qwen | 没有项目修复数据的基线 |
| PR diff only | 只看 PR/diff 弱监督能带来多少收益 |
| Rule Repair Card / no-llm | 验证规则结构化 Repair Card 的价值 |
| Full LLM Repair Card | 验证 LLM enriched root cause/rationale 的收益 |
| No-tests ablation | 验证测试证据对修复能力的贡献 |

概念上要分清楚：

- Repair SFT 是后训练 instruction 数据。
- Repair Corpus 是中训练 / continued pretraining 数据。
- 这两者都还不是 agent action-observation trajectory；真正的 agent 轨迹仍然是 runner 产生的 `trace.jsonl`。

## 5. 接真实 API 怎么理解

真实 API 模式下，模型不再用固定的 mock 逻辑，而是真的生成工具调用。

需要设置：

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="deepseek-chat"
```

然后去掉 `--mock`：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli run \
  --repo examples/buggy_calculator \
  --task "Fix the subtract function bug." \
  --test-command "python3 -m unittest discover -s tests"
```

注意：不要把 API key 写进 README、代码、trace 或 git commit。真实项目里应该用环境变量或本地 `.env`，并且不要提交 `.env`。

## 6. Docker 沙箱是什么

默认情况下，项目会把 repo 复制到：

```text
runs/<run_id>/workspace
```

然后在这个 workspace 里执行测试。

Docker 模式会进一步把测试命令放进容器里跑：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli run-manifest \
  --manifest examples/tasks.jsonl \
  --sandbox docker \
  --docker-image python:3.11-slim \
  --mock
```

Docker 模式会使用类似设置：

```text
--network none
--memory 1g
--cpus 1
--pids-limit 256
```

它的意义是：真实模型可能会生成不可预测的代码和命令，把测试执行放进受限容器里更安全。

## 7. 代码结构怎么读

建议按这个顺序看源码：

| 文件 | 作用 | 你应该重点看什么 |
|---|---|---|
| `src/agenttrace_sandbox/cli.py` | 命令行入口 | 有哪些命令，参数如何传入 |
| `src/agenttrace_sandbox/config.py` | 配置 | 环境变量如何变成配置 |
| `src/agenttrace_sandbox/runner.py` | Agent 主循环 | 模型如何一步步调用工具 |
| `src/agenttrace_sandbox/tools.py` | 工具系统 | read/grep/edit/test/diff 如何执行 |
| `src/agenttrace_sandbox/sandbox.py` | 沙箱 | local/docker backend 如何跑命令 |
| `src/agenttrace_sandbox/tracing.py` | trace 写入 | JSONL 事件如何保存 |
| `src/agenttrace_sandbox/manifest.py` | 批量运行 | JSONL 任务列表如何批量跑 |
| `src/agenttrace_sandbox/data_builders.py` | 数据构建 | benchmark/PR 记录如何变成任务或 wiki |
| `src/agenttrace_sandbox/sft_export.py` | SFT 导出 | trace 如何变成训练样本 |
| `src/agenttrace_sandbox/stats.py` | 统计 | pass rate、format violation 如何统计 |
| `src/agenttrace_sandbox/llm.py` | 模型接口 | mock 模型和真实 API 如何统一 |

## 7.1 Benchmark-to-SFT 闭环

现在项目里 benchmark 到 SFT 的完整链路是：

```text
MBPP/HumanEval 原始题目
  -> build-mbpp / build-humaneval
  -> 生成可运行 repo + tasks.jsonl
  -> run-manifest 调用真实 API 或 mock agent
  -> 生成 runs/<run_id>/trace.jsonl
  -> export-sft 导出训练样本
```

通俗地说，benchmark builder 做的是“把题目变成 agent 能操作的代码仓库”；runner 做的是“让模型真的在仓库里读文件、改代码、跑测试”；SFT exporter 做的是“把成功过程拆成下一步工具调用训练样本”。

导出高质量 SFT 数据时可以用严格模式：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli export-sft \
  --traces runs \
  --output data/sft/benchmark_tool_calls_strict.jsonl \
  --strict \
  --reject-test-edits
```

这个模式会过滤掉：

```text
没有成功完成的轨迹
没有跑通测试的轨迹
工具调用不是干净 JSON 的轨迹
发生 JSON 修复重试的轨迹
被安全策略拦截或工具报错的轨迹
改动测试文件的轨迹
```

如果模型整体成功，但中间偶尔有一步把 JSON 包进 markdown/prose，可以用更实用的 clean-step 模式：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli export-sft \
  --traces runs \
  --output data/sft/benchmark_tool_calls_clean_steps.jsonl \
  --clean-steps \
  --reject-test-edits
```

它不会因为一条轨迹里某一步不干净就丢掉整条轨迹，而是只保留成功轨迹里的干净工具调用。

查看 manifest 结果通过率：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli stats \
  --manifest-results runs/mbpp_results.jsonl
```

## 7.2 先跑多少题比较合适

建议分三档推进：

```text
Smoke:       MBPP 3-5 条 + HumanEval 3-5 条
Small eval:  MBPP 20-50 条 + HumanEval 20-50 条
Data run:    MBPP/HumanEval 全量或按预算分批跑
```

原因是公开 benchmark 题目难度不均匀。先跑 3-5 条可以检查 API、工具协议、测试执行和 SFT 导出是否正常；再跑 20-50 条才比较适合看初步成功率；全量跑更适合最后做数据集构建和论文/简历展示。

一次真实 API smoke 结果示例：

```text
MBPP:      3/3 success, pass_rate=100%, avg_steps=6.00
HumanEval: 3/3 success, pass_rate=100%, avg_steps=4.67
```

但这只是小样本，不代表全量 benchmark 的最终通过率。更有价值的是同时看：

```text
pass_rate                 任务是否做对
format_violation_rate     工具调用协议是否干净
strict SFT count          整条轨迹是否都适合直接训练
clean-step SFT count      成功轨迹中有多少干净步骤可用
```

## 8. Agent 主循环详解

核心逻辑在 `runner.py`。

简化后是：

```text
run_task()
  -> 创建 Sandbox
  -> 创建 ToolRegistry
  -> 写 run_started trace
  -> 调模型生成 plan
  -> 循环 max_steps 次：
       next_action()     # 调模型拿 JSON 工具调用
       execute_step()    # 执行工具
       write_event()     # 写 trace
       判断是否 finish / invalid_json / max_steps
  -> git_diff
  -> 写 run_finished trace
```

这里有两个关键对象：

```text
StepDecision
  raw: 模型原始输出
  action: 解析后的工具调用
  parse_error: JSON 解析错误
  retries_used: 修复重试次数
  format_violation: 是否不是纯 JSON
```

```text
ToolResult
  ok: 工具是否成功
  output: 工具输出
  blocked: 是否被安全策略拦截
  error_type: 错误类型
```

## 9. 工具调用协议

模型每一步应该输出一个 JSON：

```json
{
  "tool": "read_file",
  "arguments": {
    "path": "calculator.py"
  },
  "reason": "Inspect the implementation before editing."
}
```

支持的工具包括：

```text
list_files
read_file
grep
replace_in_file
write_file
run_tests
git_diff
finish
```

为什么要求 JSON？

因为 JSON 可以被程序稳定解析，后面也可以直接拿来做 SFT 数据。

## 10. Trace 长什么样

每次运行会生成：

```text
runs/<run_id>/trace.jsonl
```

典型事件：

```json
{"event": "run_started", "payload": {"task": "..."}}
{"event": "plan", "payload": {"plan": "..."}}
{"event": "tool_call", "payload": {"tool": "read_file", "arguments": {"path": "calculator.py"}}}
{"event": "run_finished", "payload": {"outcome": "success", "diff": "..."}}
```

真实工具调用里会记录更多字段：

```json
{
  "tool": "replace_in_file",
  "raw_output": "{...}",
  "parse_error": "",
  "retries_used": 0,
  "format_violation": false,
  "elapsed_ms": 12.3,
  "result": {
    "ok": true,
    "error_type": ""
  }
}
```

这些字段就是后训练数据和分析指标的来源。

## 11. `format_violation` 为什么重要

真实模型有时不会严格输出纯 JSON，而是输出：

````text
我先读取文件。

```json
{"tool": "read_file", "arguments": {"path": "calculator.py"}, "reason": "..."}
```
````

这种输出可以被系统“救回来”，但它不是干净的工具协议。

所以项目记录：

```json
"format_violation": true
```

这样后面可以区分：

```text
干净样本：format_violation=false 且 result.ok=true
可恢复样本：format_violation=true 但 JSON 可抽取
失败样本：parse_error 或 test_failed 或 policy_block
```

这就是一个很好的创新点：不是只记录成功，而是记录“成功质量”。

## 12. SFT 导出怎么理解

SFT 训练想教模型：

> 给定任务、计划、历史工具轨迹，下一步应该调用什么工具？

导出的样本类似：

```json
{
  "instruction": "Given a coding task, plan, and previous tool history, choose the next safe tool call as JSON.",
  "input": {
    "task": "Fix the subtract function bug.",
    "plan": "...",
    "history": []
  },
  "output": {
    "tool": "read_file",
    "arguments": {
      "path": "calculator.py"
    },
    "reason": "Inspect the implementation."
  }
}
```

也就是说，训练目标不是直接训练“最后答案”，而是训练模型学会 **下一步工具决策**。

## 13. Stats 怎么看

运行：

```bash
PYTHONPATH=src python3 -m agenttrace_sandbox.cli stats --runs runs
```

输出类似：

```json
{
  "total_runs": 14,
  "outcomes": {
    "success": 14
  },
  "pass_rate": 1.0,
  "avg_steps": 5.71,
  "avg_elapsed_ms": 4173.29,
  "format_violation_rate": 0.0,
  "backends": {
    "local": 11,
    "unknown": 3
  }
}
```

重点看：

| 字段 | 含义 |
|---|---|
| `total_runs` | 总共统计了多少次运行 |
| `outcomes` | 成功、失败、格式错误等分布 |
| `pass_rate` | 成功率 |
| `avg_steps` | 平均工具步数 |
| `avg_elapsed_ms` | 平均耗时 |
| `format_violation_rate` | 非纯 JSON 输出比例 |
| `backends` | local/docker 使用情况 |

## 14. 可以怎么讲创新点

这个项目不只是“写了一个 agent demo”，可以这样讲：

### 1. 安全轨迹采集

模型不是直接操作原仓库，而是在隔离 workspace 中执行，后续还能使用 Docker 限制网络、内存和 CPU。

### 2. 协议遵循度评估

不仅看模型有没有成功，还看它是否严格按 JSON 工具协议输出。

### 3. 成功质量分层

把轨迹分成：

```text
clean success
recovered success
test failure
policy block
invalid JSON
edit miss
```

这比单纯 pass/fail 更适合后训练数据筛选。

### 4. 最小 diff 行为分析

真实模型可能会多改测试或做额外修改。项目可以把“是否最小修改”作为后续质量评分维度。

### 5. 模型来源可追踪

trace 记录 provider、model、temperature，方便比较不同模型的轨迹质量。

### 6. 后训练数据闭环

项目可以从真实执行轨迹生成 SFT 数据，后面还能扩展到 DPO：

```text
chosen: clean/minimal success
rejected: noisy/over-editing/failure trace
```

## 15. 推荐学习路线

如果你想彻底理解这个项目，建议这样学：

1. 先跑 `--mock` demo，看一次 trace。
2. 打开 `runner.py`，理解 `run_task()` 主循环。
3. 打开 `tools.py`，理解工具如何真正改文件和跑测试。
4. 打开 `sandbox.py`，理解 local/docker 的区别。
5. 打开 `sft_export.py`，理解 trace 如何变成训练样本。
6. 打开 `stats.py`，理解如何从 trace 统计指标。
7. 最后读 `docs/REAL_API_TRACE_NOTES.md`，理解真实 API 暴露了哪些工程问题。

## 16. 下一步可以做什么

比较自然的下一步：

```text
1. strict SFT export
   只导出 format_violation=false 且 result.ok=true 的样本。

2. doctor 命令
   检查 Docker、API 环境变量、Python 版本，但不打印 key。

3. trajectory quality score
   综合 success、format_violation、test result、diff size、是否改测试。

4. 更多 toy tasks
   构造 20-50 个小任务，展示批量采集统计。

5. DPO pair export
   用 clean success 和 noisy/failure trace 构造偏好数据。
```

如果只选一个，我建议先做：

```text
strict SFT export + quality score
```

因为这能直接把“轨迹采集”升级成“高质量后训练数据构建”。

## 17. Benchmark 和 PR 数据如何接入

现在项目开始接入两个上游数据入口：

```text
benchmark/unit-test 数据 -> 可运行 repair task
GitHub PR/Issue/diff 数据 -> repair wiki 解释数据
```

这对应我们之前讨论的两条路线。

### 1. Benchmark 变成可运行任务

Benchmark 数据通常包含：

```text
函数名
题目描述
有 bug 的代码或 skeleton
单元测试
```

AgentTrace 会把它写成一个小 repo：

```text
repos/<task_id>/
  solution.py
  tests/test_solution.py
  AGENT.md
```

再写入 manifest：

```json
{
  "id": "repair_subtract_operator",
  "repo": "repos/repair_subtract_operator",
  "task": "Fix subtract so it returns a minus b.",
  "test_command": "python3 -m unittest discover -s tests"
}
```

这样它就能被 `run-manifest` 批量执行。

这条路线更接近：

```text
Unit Test Guided Repair Synthesis
```

也就是用测试反馈构造可验证的修复轨迹。

### 2. PR/Issue 变成 Repair Wiki

PR/Issue 数据通常包含：

```text
issue_title
issue_body
pr_title
pr_body
diff
files
```

AgentTrace 当前的轻量 wiki builder 会生成：

```json
{
  "bug_summary": "...",
  "change_summary": "...",
  "fix_strategy": "...",
  "validation": "..."
}
```

它不是直接跑 agent，而是生成代码理解/修复解释数据。

这条路线更接近：

```text
PR/Issue Wiki mid-training-style data
```

注意：现在只是轻量版数据构建，还不是大规模 mid-training。

### 3. 两条路线如何汇合

后续可以这样汇合：

```text
Benchmark task
  -> Agent 执行
  -> trace
  -> SFT/DPO

PR/Issue Wiki
  -> bug summary / root cause / fix strategy
  -> plan/reasoning 数据
  -> mid-training-style 代码理解语料
```

这让项目从“只跑一个 demo”变成：

```text
上游数据构造
  -> 安全执行验证
  -> 轨迹质量分析
  -> 后训练数据导出
```
