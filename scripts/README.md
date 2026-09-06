# 调用方式

当前项目的原始数据位于项目根目录 `raw/`。标准化 JSONL 位于 `standardized/`。

从项目根目录运行：

```bash
conda run -n shine python scripts/organize_datasets.py
```

脚本会从 `raw/` 读取原始 JSON、CSV、Parquet 和 RACE 文件，并重建 `standardized/`。

NarrativeQA 完整故事下载与单独标准化：

```bash
NARRATIVEQA_SLEEP_SECONDS=0 bash raw/narrativeqa/download_stories.sh
conda run -n shine python scripts/organize_datasets.py --datasets narrativeqa
```

standardize 会读取 `raw/narrativeqa/tmp/*.content`；`invalid_downloads/` 中的隔离文件不会被读取，相关样本自动使用官方 summary。

统计 context token 规模（Qwen tokenizer 抽样估计）：

```bash
conda run -n shine python scripts/context_stats_approx.py
```

按 context 聚合 QA：

```bash
conda run -n shine python scripts/aggregate_contexts.py
```

结果写入 `aggregated/<dataset>/contexts.jsonl`；MS MARCO v1.1/v2.1 使用共同的 `aggregated/ms_marco/` 命名空间。

将聚合结果按 `train`、`validation`、`test` 输出：

```bash
conda run -n shine python scripts/split_aggregated.py
```

该命令在每个 `aggregated/<dataset>/` 下生成三个 JSONL 视图，并保留原始 `contexts.jsonl`。跨 split context 重复会记录在 `split_aggregation_stats.json`。

科学论文 QA（Qasper、SciDQA）标准化：

```bash
conda run -n shine python scripts/organize_scientific_qa.py
```

PeerQA 官方数据包下载与转换（源站可访问时）：

```bash
bash scripts/download_peerqa.sh
conda run -n shine python scripts/organize_peerqa.py
```
