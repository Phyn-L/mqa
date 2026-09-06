# QA 生成工具

`qa_gen/` 将带有上下文的 JSONL 数据转换为可追溯的多轮问答数据。流水线分为三个阶段：

```text
contexts.jsonl
    │
    ├─ make_pilot.py              选择小规模试验集（可选）
    ├─ fact_extraction.py         抽取并校验原子事实
    ├─ multiturn_qa_generation.py 生成多轮对话并做确定性校验
    └─ multiturn_qa_audit.py      LLM 审计，失败时可重生成
```

模型默认使用 `Qwen/Qwen3.5-9B`，通过 Transformers 的 `AutoProcessor` 和
`AutoModelForMultimodalLM` 加载。运行前请在项目环境中安装 PyTorch、
Transformers，并确保模型可从本地缓存或模型仓库加载。

## 目录内容

- `make_pilot.py`：从 `aggregated/` 中按数据集抽取短上下文，生成试验输入。
- `fact_extraction.py`：从上下文抽取原子事实，检查 JSON、事实 ID、证据子串和重要性分数。
- `multiturn_qa_generation.py`：读取事实结果，生成带对话依赖的多轮 QA，并检查来源、事实、证据和历史依赖。
- `multiturn_qa_audit.py`：对 QA 做程序化检查和 LLM 整体判定，可对失败候选进行有限次修复。
- `utils.py`：JSONL 读取、JSON 容错解析、左填充、生成结果解码和 sortish batching 等公共函数。
- `prompts/`：三个阶段使用的 prompt 模板。模板中的占位符由脚本填充，不建议在命令行中直接拼接上下文。

## 输入约定

所有主流程输入均为 UTF-8 编码的 JSONL，每行一个对象，至少包含：

```json
{
  "context_id": "ctx-001",
  "context": "原始上下文文本",
  "dataset": "squad",
  "split": "train"
}
```

`context` 必须是非空字符串。`context_id` 缺失时，部分脚本会回退到 `id`；建议始终显式提供
`context_id`。事实抽取阶段会保留 `context_id`、数据集来源、split 和完整上下文，后续阶段将这些字段视为可信来源，不接受模型修改。

## 快速开始

以下命令均从项目根目录 `/data/lz/mqa` 执行。默认 `--max-samples=2` 只处理两条记录，适合先验证流程；批量运行时请显式设置样本数。

### 1. 生成 pilot 输入（可选）

```bash
python qa_gen/make_pilot.py \\
  --root aggregated \\
  --per-dataset 3 \\
  --max-chars 4000 \\
  --output qa_gen/pilot_contexts.jsonl
```

脚本会尝试读取 `squad`、`coqa`、`drop`、`qasper`、`narrativeqa`、`pwc`、`race`、`quac`
下的 `contexts.jsonl`，跳过不存在的文件、过短文本和超过 `--max-chars` 的文本。

### 2. 抽取原子事实

```bash
python qa_gen/fact_extraction.py \\
  --input qa_gen/pilot_contexts.jsonl \\
  --prompt qa_gen/prompts/fact_extract_prompt.txt \\
  --output-dir qa_gen/generated_fact_extraction_small \\
  --model Qwen/Qwen3.5-9B \\
  --max-samples 24 \\
  --batch-size 1
```

首轮生成是确定性的；失败记录按 `--max-attempts`（默认 3）进行重试，重试可使用采样参数
`--retry-temperature`、`--retry-top-p` 和 `--retry-top-k`。`--token-budget` 可限制一个 batch
中（最长输入 + 预留生成长度）的总 token 数。

输出目录包含：

- `facts.jsonl`：通过校验的事实记录；
- `failed.jsonl`：未通过记录、错误原因、原始模型输出和尝试次数；
- `summary.json`：总数、有效 JSON 数、写入数、拒绝数及错误计数。

事实记录的核心结构为：

```json
{"context_id": "ctx-001", "source": {"dataset": "squad", "split": "train"},
 "context": "完整原文", "facts": [{"fact_id": "F1", "text": "...",
 "fact_type": "event", "importance": 0.8, "evidence": "原文连续片段"}]}
```

每条事实必须有唯一 `fact_id`、非空文本、`importance`（`[0,1]`）以及一个在原文中逐字符出现的连续 `evidence` 字符串。每个原子事实只表达一个独立、可验证的 claim，并由 context 中一个连续片段直接支持；不跨句拼接、不依赖多个分散位置、不进行跨句推理或事实组合。

### 3. 生成多轮 QA

```bash
python qa_gen/multiturn_qa_generation.py \\
  --input qa_gen/generated_fact_extraction_small/facts.jsonl \\
  --prompt qa_gen/prompts/multiturn_qa_prompt.txt \\
  --output-dir qa_gen/generated_multiturn_qa \\
  --model Qwen/Qwen3.5-9B \\
  --max-samples 24 \\
  --batch-size 1
```

生成器要求输出 `dialogue_plan` 和 `conversation`。每个 turn 至少包含 `turn_id`、
`question`、`answer`、`required_facts`、`depends_on_turns`、`requires_history` 和 `evidence`。
校验器还会检查：上下文、来源和事实是否被修改；事实 ID 是否存在；证据是否为上下文子串；依赖是否只指向更早 turn；首 turn 是否错误依赖历史；以及对话是否覆盖至少一半原子事实。

输出目录包含：

- `qa.jsonl`：通过确定性校验的对话；
- `rejected.jsonl`：被拒绝的上下文及错误信息；
- `summary.json`：总数、接受数和拒绝数。

### 4. 审计和修复

```bash
python qa_gen/multiturn_qa_audit.py \\
  --input qa_gen/generated_multiturn_qa/qa.jsonl \\
  --prompt qa_gen/prompts/multiturn_qa_audit_prompt.txt \\
  --generation-prompt qa_gen/prompts/multiturn_qa_prompt.txt \\
  --output-dir qa_gen/multiturn_qa_audit \\
  --model Qwen/Qwen3.5-9B \\
  --max-samples 24 \\
  --max-regenerations 2
```

审计先运行与生成阶段相同的确定性校验，再要求模型返回 `{"valid": true}` 或带 `reason`
的 `{"valid": false}`。失败候选最多按 `--max-regenerations` 修复（默认 2）；修复时会重新生成完整 JSON，而不是拼接补丁。

输出目录包含：

- `audit.jsonl`：每条记录的确定性错误、LLM 判定、接受状态和尝试历史；
- `accepted.jsonl`：最终通过审计的 QA；
- `rejected.jsonl`：未通过审计的原始记录；
- `summary.json`：总数、接受数、拒绝数和发生重生成的记录数。

## 常用参数

三个模型脚本共享以下运行参数：

| 参数 | 作用 | 默认值 |
| --- | --- | --- |
| `--input` | 输入 JSONL | 各脚本的 `DEFAULT_INPUT` |
| `--output-dir` | 输出目录 | 各脚本的 `DEFAULT_OUTPUT_DIR` |
| `--model` | Hugging Face 模型名或本地路径 | `Qwen/Qwen3.5-9B` |
| `--max-samples` | 最多处理的记录数 | `2` |
| `--batch-size` | 推理 batch 大小 | `1` |
| `--max-new-tokens` | 首轮最大生成长度 | 抽取 `32768`；QA 生成 `32768`；审计 `8192` |
| `--sortish-window-size` | 局部长度排序窗口 | `2000` |
| `--sortish-seed` | batching 随机种子 | `42` |
| `--token-budget` | 可选的 batch token 上限 | 不限制 |
| `--device-map` | Transformers 设备映射 | `auto` |
| `--torch-dtype` | 模型 dtype | `auto` |

抽取脚本额外提供 `--retry-max-new-tokens`、`--max-attempts` 和 retry sampling 参数；审计脚本额外提供 `--generation-prompt` 与 `--max-regenerations`。完整参数以 `python <script> --help` 为准。

## SQuAD/CoQA 夜间全流程

使用 `run_squad_coqa_pipeline.sh` 可依次处理 SQuAD v1.1 train 和 CoQA train：事实抽取、
多轮 QA 生成、LLM 审计与失败重生成。每个数据集写入独立的 `facts/`、`qa/`、`audit/`
目录和 `pipeline.log`。阶段成功后会生成 `summary.json`；再次启动时会跳过已完成阶段。

夜间全量运行：

```bash
mkdir -p qa_gen/nightly_squad_coqa
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash qa_gen/run_squad_coqa_pipeline.sh \
  > qa_gen/nightly_squad_coqa/nohup.log 2>&1 &
echo $! > qa_gen/nightly_squad_coqa/pipeline.pid
```

建议先做小规模试跑：

```bash
MAX_SAMPLES=20 bash qa_gen/run_squad_coqa_pipeline.sh
```

可用环境变量：`MAX_SAMPLES`、`OUTPUT_ROOT`、`MODEL`、`BATCH_SIZE`、`TOKEN_BUDGET`、
`FACT_MAX_NEW_TOKENS`、`QA_MAX_NEW_TOKENS`、`AUDIT_MAX_NEW_TOKENS`、`FACT_MAX_ATTEMPTS`、
`MAX_REGENERATIONS` 和 `FORCE_RERUN=1`。

查看进度：

```bash
tail -f qa_gen/nightly_squad_coqa/squad/pipeline.log
tail -f qa_gen/nightly_squad_coqa/coqa/pipeline.log
```

## 可复现性与排错

- 固定 `--sortish-seed`，并记录模型、prompt 文件版本、命令行参数和输出目录。
- 使用 `--batch-size 1 --max-samples 1` 先做端到端冒烟测试，再扩大规模。
- 若出现显存不足，降低 `--batch-size`、设置更小的 `--token-budget` 或减少 `--max-new-tokens`。
- `failed.jsonl`、`rejected.jsonl` 和 `audit.jsonl` 中保留失败原因及原始输出，应在质量分析时一并检查，不能静默丢弃。
- 生成阶段依赖左填充和仅解码新生成 token；不要在公共工具中改回右填充。

脚本只负责生成和校验，不会自动修改 `standardized/` 或 `aggregated/` 中的源数据。
