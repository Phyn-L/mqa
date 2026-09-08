"""Stage 1: extract atomic facts with incremental success/failure output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


from utils import (
    IncrementalJsonlWriter,
    ModelRunner,
    context_id,
    fill_prompt,
    parse_json_object,
    read_jsonl,
)

DEFAULT_PROMPT = (
    Path(__file__).with_name("prompts") / "fact_extract_with_candidates_prompt.txt"
)


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
    output = args.output_dir / "facts.jsonl"
    failed_output = args.output_dir / "failed.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if output.exists():
        completed = index_by_context_id(read_jsonl(output), "facts output")
        for key, record in completed.items():
            if not isinstance(record.get("facts"), list):
                raise ValueError(f"Existing facts record {key} has no facts list")
    print(f"{len(completed)}/{len(candidate_records)} records has been processed")

    ids: list[str] = []
    texts: list[str] = []
    candidates: list[Any] = []
    for index, candidate_record in enumerate(candidate_records):
        key = context_id(candidate_record, index)
        if key in completed:
            continue
        if key not in contexts:
            raise ValueError(f"candidates references unknown context_id {key}")
        text = contexts[key].get("context")
        candidate_payload = candidate_record.get("candidates")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Context {key} has no non-empty context")
        if not (
            isinstance(candidate_payload, dict)
            or (isinstance(candidate_payload, str) and candidate_payload.strip())
        ):
            raise ValueError(f"Candidate record {key} has no candidates text/object")
        ids.append(key)
        texts.append(text)
        candidates.append(candidate_payload)
    template = args.prompt.read_text(encoding="utf-8")
    prompts = [
        fill_prompt(
            template,
            {
                "{{CONTEXT}}": text,
                "{{CANDIDATES}}": (
                    candidate
                    if isinstance(candidate, str)
                    else json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
                ),
            },
        )
        for text, candidate in zip(texts, candidates)
    ]
    writer = IncrementalJsonlWriter(output, failed_output)
    with writer:
        if prompts:
            runner = ModelRunner(args.model, args.device_map, args.torch_dtype)

            def write_result(index: int, raw: str) -> None:
                key = ids[index]
                parsed, parse_error = parse_json_object(raw)
                errors = [parse_error] if parse_error else validate_facts(
                    parsed, texts[index]
                )
                if errors:
                    writer.write_failure(key, raw)
                else:
                    writer.write_success(
                        {"context_id": key, "facts": parsed["facts"]}
                    )

            runner.generate(
                prompts,
                args.batch_size,
                args.max_new_tokens,
                sortish_window_size=args.sortish_window_size,
                sortish_seed=args.sortish_seed,
                token_budget=args.token_budget,
                progress_desc="fact extraction",
                result_callback=write_result,
            )

    print(
        f"existing={len(completed)} written={writer.written} failed={writer.failed} "
        f"output={output} failed_output={failed_output}"
    )
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
    parser.add_argument("--sortish-window-size", type=int, default=2000)
    parser.add_argument("--sortish-seed", type=int, default=42)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
