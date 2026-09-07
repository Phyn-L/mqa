"""Stage 1: extract atomic facts and write only facts.jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .utils import (
        ModelRunner, context_id, fill_prompt, parse_json_object, read_jsonl,
        write_jsonl,
    )
except ImportError:
    from utils import (
        ModelRunner, context_id, fill_prompt, parse_json_object, read_jsonl,
        write_jsonl,
    )


DEFAULT_PROMPT = Path(__file__).with_name("prompts") / "fact_extract_with_candidates_prompt.txt"


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


def validate_facts(parsed: dict[str, Any] | None, context: str) -> list[str]:
    if parsed is None:
        return ["missing JSON object"]
    if set(parsed) != {"facts"}:
        return ["top-level output must contain only facts"]
    facts = parsed["facts"]
    if not isinstance(facts, list):
        return ["facts must be a list"]
    errors: list[str] = []
    ids: set[str] = set()
    for index, fact in enumerate(facts):
        prefix = f"facts[{index}]"
        if not isinstance(fact, dict):
            errors.append(f"{prefix} must be an object")
            continue
        fact_id = fact.get("fact_id")
        if not isinstance(fact_id, str) or not fact_id:
            errors.append(f"{prefix}.fact_id must be non-empty")
        elif fact_id in ids:
            errors.append(f"duplicate fact_id {fact_id}")
        else:
            ids.add(fact_id)
        if not isinstance(fact.get("text"), str) or not fact["text"].strip():
            errors.append(f"{prefix}.text must be non-empty")
        evidence = fact.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            errors.append(f"{prefix}.evidence must be non-empty")
        elif evidence not in context:
            errors.append(f"{prefix}.evidence is not a context substring")
        importance = fact.get("importance")
        if (
            not isinstance(importance, (int, float))
            or isinstance(importance, bool)
            or not 0 <= importance <= 1
        ):
            errors.append(f"{prefix}.importance must be in [0, 1]")
    return errors


def run(args: argparse.Namespace) -> Path:
    contexts = index_by_context_id(read_jsonl(args.contexts), "contexts")
    candidate_records = read_jsonl(args.candidates, args.max_samples)
    ids: list[str] = []
    texts: list[str] = []
    candidates: list[dict[str, Any]] = []
    for index, candidate_record in enumerate(candidate_records):
        key = context_id(candidate_record, index)
        if key not in contexts:
            raise ValueError(f"candidates references unknown context_id {key}")
        text = contexts[key].get("context")
        candidate_payload = candidate_record.get("candidates")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Context {key} has no non-empty context")
        if not isinstance(candidate_payload, dict):
            raise ValueError(f"Candidate record {key} has no candidates object")
        ids.append(key)
        texts.append(text)
        candidates.append(candidate_payload)
    template = args.prompt.read_text(encoding="utf-8")
    prompts = [
        fill_prompt(template, {
            "{{CONTEXT}}": text,
            "{{CANDIDATES_JSON}}": json.dumps(
                candidate, ensure_ascii=False, separators=(",", ":")
            ),
        })
        for text, candidate in zip(texts, candidates)
    ]
    runner = ModelRunner(args.model, args.device_map, args.torch_dtype)
    raw_outputs = runner.generate(
        prompts, args.batch_size, args.max_new_tokens,
        sortish_window_size=args.sortish_window_size,
        sortish_seed=args.sortish_seed, token_budget=args.token_budget,
    )

    output_records: list[dict[str, Any]] = []
    skipped = 0
    for key, text, prompt, raw in zip(ids, texts, prompts, raw_outputs):
        parsed, parse_error = parse_json_object(raw)
        errors = [parse_error] if parse_error else validate_facts(parsed, text)
        attempts = 1
        while errors and attempts < args.max_attempts:
            raw = runner.generate(
                [prompt], 1, args.max_new_tokens, sample=True,
                sortish_window_size=args.sortish_window_size,
                sortish_seed=args.sortish_seed, token_budget=args.token_budget,
            )[0]
            parsed, parse_error = parse_json_object(raw)
            errors = [parse_error] if parse_error else validate_facts(parsed, text)
            attempts += 1
        if errors:
            skipped += 1
            continue
        output_records.append({"context_id": key, "facts": parsed["facts"]})

    output = args.output_dir / "facts.jsonl"
    write_jsonl(output, output_records)
    print(f"written={len(output_records)} skipped={skipped} output={output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--sortish-window-size", type=int, default=2000)
    parser.add_argument("--sortish-seed", type=int, default=42)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
