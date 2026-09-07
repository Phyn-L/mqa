# 独立分阶段 QA 数据生成

每个脚本只执行一个阶段。`context_id` 只用于关联记录，各 JSONL 不复制其他阶段的数据。候选处理明确拆成两个阶段：先完整运行 spaCy/规则并落盘，再由另一个脚本读取完整候选并调用 LLM。

完整顺序为：

```text
contexts.jsonl
  -> extract_spacy_candidates.py
  -> spacy_candidates.jsonl
  -> reorganize_candidates.py
  -> candidates.jsonl
  -> extract_facts.py
  -> facts.jsonl
  -> generate_qa.py
  -> qa.jsonl
  -> validate_qa.py
  -> rejected.jsonl
```

三个阶段都支持 sortish batching 参数：`--batch-size`、`--sortish-window-size`、`--sortish-seed` 和可选的 `--token-budget`。工具会按 chat-template 后的 prompt 长度局部排序，在 batch 内减少 padding，并按原始输入顺序写回结果。

`compare_spacy_models.py` 可对同一文本比较已安装的 spaCy pipeline：

```bash
conda run -n shine python qa_gen/staged_pipeline/compare_spacy_models.py \
  --input qa_gen/staged_pipeline/examples/french_indian_war.txt \
  --output /tmp/spacy_comparison.json \
  --models en_core_web_sm en_core_web_trf
```

实测比较见 `examples/french_indian_war_comparison.md`，最终压缩结构见 `examples/french_indian_war_candidates.json.example`。

## 0a. spaCy/规则候选抽取

该阶段不加载 LLM，只批量运行 spaCy 和规则，直到所有 context 处理完成后写出 `spacy_candidates.jsonl`：

```bash
conda run -n shine python qa_gen/staged_pipeline/extract_spacy_candidates.py \
  --input /path/to/contexts.jsonl \
  --output-dir /path/to/output/spacy_candidates \
  --spacy-model en_core_web_sm \
  --batch-size 128
```

每行包含 `context_id` 和高召回的 `candidates`，包括 entities、numeric_temporal、triggers、relations 和 priority_sentences。
该文件只保存 spaCy/规则结果，不包含 LLM 语义重组结果。

## 0b. LLM 候选语义重组

该阶段只读取原始 contexts 与已经完整生成的 `spacy_candidates.jsonl`，不再运行 spaCy：

```bash
conda run -n shine python qa_gen/staged_pipeline/reorganize_spacy_candidates.py \
  --contexts /path/to/contexts.jsonl \
  --spacy-candidates /path/to/output/spacy_candidates/spacy_candidates.jsonl \
  --output-dir /path/to/output/candidates \
  --model /path/to/model \
  --min-coverage 1.0
```

该阶段输出唯一文件 `candidates.jsonl`。每条 LLM 结果必须通过 JSON schema、所有 evidence 原文 substring 和逐类别候选覆盖率校验。
主批量生成按 sortish batching 后的实际 batch 数显示 `LLM reorganization` 进度条；单条失败重试不重复显示进度条。

## 1. Atomic fact 抽取

```bash
conda run -n shine python qa_gen/staged_pipeline/extract_facts.py \
  --contexts /path/to/contexts.jsonl \
  --candidates /path/to/output/candidates/candidates.jsonl \
  --output-dir /path/to/output/facts \
  --model /path/to/model
```

唯一输出 `facts.jsonl`：

```json
{"context_id": "ctx-001", "facts": [{"fact_id": "F1", "text": "...", "evidence": "...", "importance": 0.8}]}
```

## 2. Multi-turn QA 生成

```bash
conda run -n shine python qa_gen/staged_pipeline/generate_qa.py \
  --contexts /path/to/contexts.jsonl \
  --facts /path/to/output/facts/facts.jsonl \
  --output-dir /path/to/output/qa \
  --model /path/to/model
```

唯一输出 `qa.jsonl`。它不保存 context 或 facts，只保存 QA 和 evidence：

```json
{"context_id": "ctx-001", "qa": {"dialogue_plan": [], "conversation": [{"turn_id": "T1", "question": "...", "answer": "..."}]}, "evidence": [{"turn_id": "T1", "spans": ["原文片段"]}]}
```

## 3. QA 校验

```bash
conda run -n shine python qa_gen/staged_pipeline/validate_qa.py \
  --contexts /path/to/contexts.jsonl \
  --facts /path/to/output/facts/facts.jsonl \
  --qa /path/to/output/qa/qa.jsonl \
  --output-dir /path/to/output/validation \
  --model /path/to/model
```

唯一输出 `rejected.jsonl`，每行只包含未通过的 context 编号：

```json
{"context_id": "ctx-001"}
```

程序校验失败、缺失 facts/QA/evidence 或 LLM 审计失败都会被记为未通过。
