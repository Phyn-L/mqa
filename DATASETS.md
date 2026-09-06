# Dataset organization

`raw/` stores downloaded source files. `standardized/` stores UTF-8 JSONL records with the common fields `id`, `context`, `question`, `answers`, `answer_starts`, `metadata`, `dataset`, and `split`. The original local HF parquet files are preserved in place.

Run the conversion again with:

```bash
conda run -n shine python organize_datasets.py
```

The authoritative source/status list is `dataset_manifest.json`. `answer_starts` is empty for datasets whose source format does not expose character offsets. For MS MARCO, `context` contains the selected passage(s), while all retrieved passages are retained under `metadata.passages`.

## Context token length distribution

For large-scale QA generation, context lengths were measured with the local Qwen3.5-9B tokenizer. The length includes the fixed prompt in `qa_gen/fact_extract_prompt.txt`; statistics cover 491,288 records in `aggregated/*/contexts.jsonl`.

| Dataset | Contexts | p50 tokens | p90 tokens | p95 tokens | Max tokens |
|---|---:|---:|---:|---:|---:|
| coqa | 7,070 | 923 | 1,004 | 1,038 | 1,891 |
| drop | 6,108 | 840 | 1,026 | 1,158 | 2,870 |
| duorc | 14,240 | 1,385 | 2,525 | 3,330 | 16,041 |
| ms_marco | 410,733 | 670 | 1,280 | 1,360 | 3,035 |
| narrativeqa | 1,572 | 45,445 | 167,930 | 239,319 | 506,738 |
| pwc | 17,606 | 1,044 | 1,136 | 1,152 | 1,487 |
| qasper | 1,585 | 5,518 | 8,569 | 10,197 | 36,469 |
| quac | 7,843 | 1,086 | 1,290 | 1,380 | 2,982 |
| race | 2,707 | 996 | 1,255 | 1,399 | 2,040 |
| scidqa | 727 | 19,493 | 45,634 | 58,546 | 119,051 |
| squad | 21,097 | 716 | 816 | 857 | 1,546 |
| **All** | **491,288** | **688** | **1,296** | **1,423** | **506,738** |

The distribution is strongly long-tailed. Most datasets have a median around 0.7k–1.4k tokens, whereas NarrativeQA, SciDQA, and QASPER contain very long contexts. Contexts exceeding the model's maximum window should be split by story/section or processed with a sliding window before generation.

Length-aware batching is recommended: sort or bucket by token length and use a dynamic token budget that accounts for both input length and `max_new_tokens`. In the full aggregate, estimated input-token utilization for fixed batches of 4/8/16 improves from 65.6%/53.5%/45.2% in file order to approximately 99.9%/99.7%/99.3% under ideal length sorting.
