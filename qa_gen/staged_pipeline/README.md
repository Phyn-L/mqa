# 独立分阶段 QA 数据生成

每个脚本只执行一个阶段。`context_id` 只用于关联记录，各 JSONL 不复制其他阶段的数据。当前只实现 atomic facts。

## 1. Atomic fact 抽取

```bash
conda run -n shine python qa_gen/staged_pipeline/extract_facts.py \
  --input /path/to/contexts.jsonl \
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
