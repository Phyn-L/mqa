"""Prepare two distinct SQuAD contexts for the QA-generation smoke test."""
from __future__ import annotations

import json
from pathlib import Path


INPUT = Path("standardized/squad/train-v1.1.jsonl")
OUTPUT = Path("qa_gen/smoke_20260906/contexts.jsonl")


def main() -> None:
    selected: list[dict] = []
    seen_contexts: set[str] = set()
    with INPUT.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            context = record["context"]
            if context in seen_contexts:
                continue
            seen_contexts.add(context)
            selected.append(
                {
                    "context_id": record["id"],
                    "context": context,
                    "dataset": record["dataset"],
                    "split": record["split"],
                }
            )
            if len(selected) == 2:
                break
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "records": len(selected)}))


if __name__ == "__main__":
    main()
