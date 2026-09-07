"""Stage 1: extract atomic facts and write only facts.jsonl."""
from __future__ import annotations

import argparse
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


DEFAULT_PROMPT = Path(__file__).resolve().parents[1] / "prompts/fact_extract_prompt.txt"


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
    records = read_jsonl(args.input, args.max_samples)
    ids = [context_id(record, index) for index, record in enumerate(records)]
    if len(ids) != len(set(ids)):
        raise ValueError("input contains duplicate context_id values")
    for index, record in enumerate(records):
        if not isinstance(record.get("context"), str) or not record["context"].strip():
            raise ValueError(f"Record {index} has no non-empty context")

    template = args.prompt.read_text(encoding="utf-8")
    prompts = [fill_prompt(template, {"{{CONTEXT}}": r["context"]}) for r in records]
    runner = ModelRunner(args.model, args.device_map, args.torch_dtype)
    raw_outputs = runner.generate(prompts, args.batch_size, args.max_new_tokens)

    output_records: list[dict[str, Any]] = []
    skipped = 0
    for key, record, prompt, raw in zip(ids, records, prompts, raw_outputs):
        parsed, parse_error = parse_json_object(raw)
        errors = [parse_error] if parse_error else validate_facts(parsed, record["context"])
        attempts = 1
        while errors and attempts < args.max_attempts:
            raw = runner.generate([prompt], 1, args.max_new_tokens, sample=True)[0]
            parsed, parse_error = parse_json_object(raw)
            errors = [parse_error] if parse_error else validate_facts(parsed, record["context"])
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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
