# QA 数据集

项目只保留两类活动内容：

- `raw/`：原始数据文件；
- `standardized/`：后续 QA 生成直接使用的统一 JSONL。

数据转换调用方式见 [`scripts/README.md`](scripts/README.md)，入口脚本为 [`scripts/organize_datasets.py`](scripts/organize_datasets.py)。HF 缓存和旧元数据已移至 `archive/hf_residue/`，不会参与运行。

原始数据说明：[`raw/`](raw/)。
