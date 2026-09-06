from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1] / "aggregated")
    parser.add_argument("--per-dataset", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("pilot_contexts.jsonl"))
    args = parser.parse_args()
    # Cover short QA passages, conversational histories, numerical reasoning,
    # scientific text, narrative, and reports.
    datasets = ["squad", "coqa", "drop", "qasper", "narrativeqa", "pwc", "race", "quac"]
    selected: list[dict] = []
    for dataset in datasets:
        path = args.root / dataset / "contexts.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            count = 0
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                context = record.get("context")
                if (not isinstance(context, str) or len(context.strip()) < 40
                        or len(context) > args.max_chars):
                    continue
                selected.append({
                    "id": record.get("context_id"),
                    "context_id": record.get("context_id"),
                    "context": context,
                    "dataset": dataset,
                    "split": "pilot",
                })
                count += 1
                if count >= args.per_dataset:
                    break
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(selected), "datasets": sorted({r["dataset"] for r in selected})}))


if __name__ == "__main__":
    main()
