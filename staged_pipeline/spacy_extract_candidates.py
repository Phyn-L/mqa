"""Stage 0a: batch-extract high-recall candidates with spaCy and rules."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import spacy

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - keeps the stage usable without tqdm
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from .candidate_utils import extract_parser_candidates
    from .utils import context_id, read_jsonl, write_jsonl
except ImportError:
    from candidate_utils import extract_parser_candidates
    from utils import context_id, read_jsonl, write_jsonl


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    for index, record in enumerate(records):
        ids.append(context_id(record, index))
        text = record.get("context")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Record {index} has no non-empty context")
        texts.append(text)
    if len(ids) != len(set(ids)):
        raise ValueError("input contains duplicate context_id values")
    return ids, texts


def run(args: argparse.Namespace) -> Path:
    if args.batch_size < 1 or args.processes < 1:
        raise ValueError("batch_size and processes must be positive")
    input = Path.joinpath(args.input, args.dataset, "contexts.jsonl")
    records = read_jsonl(input, args.max_samples)
    ids, texts = validate_records(records)
    nlp = spacy.load(args.spacy_model)
    output_records: list[dict[str, Any]] = []
    docs = nlp.pipe(texts, batch_size=args.batch_size, n_process=args.processes)
    for index, doc in enumerate(
            tqdm(docs, total=len(texts), desc="spaCy candidates", unit="context")
    ):
        output_records.append({
            "context_id": ids[index],
            "candidates": extract_parser_candidates(doc),
        })
    output = Path.joinpath(args.output_dir, args.dataset, "spacy_candidates.jsonl")
    write_jsonl(output, output_records)
    print(f"written={len(output_records)} output={output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--input", type=Path, default=Path("/data/lz/contexts/aggregated"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/lz/contexts/spacy_candidates"))
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
