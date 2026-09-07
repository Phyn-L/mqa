"""Stage 0b: reorganize completed spaCy candidates with an LLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .candidate_utils import compact_parser_candidates, validate_reorganized
    from .utils import (
        ModelRunner, context_id, fill_prompt, parse_json_object, read_jsonl,
        write_jsonl,
    )
except ImportError:
    from candidate_utils import compact_parser_candidates, validate_reorganized
    from utils import (
        ModelRunner, context_id, fill_prompt, parse_json_object, read_jsonl,
        write_jsonl,
    )


DEFAULT_PROMPT = Path(__file__).with_name("prompts") / "candidate_reorganization_prompt.txt"


def index_by_context_id(
    records: list[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        key = context_id(record, index)
        if key in indexed:
            raise ValueError(f"{label} contains duplicate context_id {key}")
        indexed[key] = record
    return indexed


def run(args: argparse.Namespace) -> Path:
    if not 0 <= args.min_coverage <= 1:
        raise ValueError("min_coverage must be in [0, 1]")
    contexts = index_by_context_id(read_jsonl(args.contexts), "contexts")
    spacy_records = read_jsonl(args.spacy_candidates, args.max_samples)
    ids: list[str] = []
    texts: list[str] = []
    seeds: list[dict[str, Any]] = []
    for index, record in enumerate(spacy_records):
        key = context_id(record, index)
        if key not in contexts:
            raise ValueError(f"spaCy candidates references unknown context_id {key}")
        text = contexts[key].get("context")
        candidate_payload = record.get("candidates")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Context {key} has no non-empty context")
        if not isinstance(candidate_payload, dict):
            raise ValueError(f"spaCy record {key} has no candidates object")
        ids.append(key)
        texts.append(text)
        seeds.append(candidate_payload)

    template = args.prompt.read_text(encoding="utf-8")
    prompts = [
        fill_prompt(template, {
            "{{CONTEXT}}": text,
            "{{PARSER_CANDIDATES_JSON}}": json.dumps(
                compact_parser_candidates(candidate), ensure_ascii=False,
                separators=(",", ":"),
            ),
        })
        for text, candidate in zip(texts, seeds)
    ]
    runner = ModelRunner(args.model, args.device_map, args.torch_dtype)
    raw_outputs = runner.generate(
        prompts, args.batch_size, args.max_new_tokens,
        sortish_window_size=args.sortish_window_size,
        sortish_seed=args.sortish_seed, token_budget=args.token_budget,
        progress_desc="LLM reorganization",
    )

    accepted: list[dict[str, Any]] = []
    skipped = 0
    for key, text, seed, prompt, raw in zip(ids, texts, seeds, prompts, raw_outputs):
        parsed, parse_error = parse_json_object(raw)
        errors, _ = (
            ([parse_error], {}) if parse_error
            else validate_reorganized(parsed, text, seed, args.min_coverage)
        )
        attempts = 1
        while errors and attempts < args.max_attempts:
            raw = runner.generate(
                [prompt], 1, args.max_new_tokens, sample=True,
                sortish_window_size=args.sortish_window_size,
                sortish_seed=args.sortish_seed, token_budget=args.token_budget,
            )[0]
            parsed, parse_error = parse_json_object(raw)
            errors, _ = (
                ([parse_error], {}) if parse_error
                else validate_reorganized(parsed, text, seed, args.min_coverage)
            )
            attempts += 1
        if errors:
            skipped += 1
            continue
        accepted.append({"context_id": key, "candidates": parsed})

    output = args.output_dir / "candidates.jsonl"
    write_jsonl(output, accepted)
    print(f"written={len(accepted)} skipped={skipped} output={output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--spacy-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument("--sortish-window-size", type=int, default=2000)
    parser.add_argument("--sortish-seed", type=int, default=42)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
