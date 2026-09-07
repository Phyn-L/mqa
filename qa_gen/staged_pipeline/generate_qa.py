"""Stage 2: generate QA and evidence into qa.jsonl."""
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


DEFAULT_PROMPT = Path(__file__).resolve().parents[1] / "prompts/multiturn_qa_prompt.txt"


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


def normalize_output(
    parsed: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    if not isinstance(parsed, dict):
        return None
    plan = parsed.get("dialogue_plan")
    conversation = parsed.get("conversation")
    if not isinstance(plan, list) or not isinstance(conversation, list) or not conversation:
        return None
    qa_turns: list[dict[str, Any]] = []
    evidence_turns: list[dict[str, Any]] = []
    for turn in conversation:
        if not isinstance(turn, dict) or not isinstance(turn.get("turn_id"), str):
            return None
        qa_turn = dict(turn)
        evidence = qa_turn.pop("evidence", None)
        if not isinstance(evidence, list) or not all(isinstance(span, str) for span in evidence):
            return None
        qa_turns.append(qa_turn)
        evidence_turns.append({"turn_id": turn["turn_id"], "spans": evidence})
    return {"dialogue_plan": plan, "conversation": qa_turns}, evidence_turns


def run(args: argparse.Namespace) -> Path:
    contexts = index_by_context_id(read_jsonl(args.contexts), "contexts")
    fact_records = read_jsonl(args.facts, args.max_samples)
    prompts: list[str] = []
    keys: list[str] = []
    template = args.prompt.read_text(encoding="utf-8")
    for index, fact_record in enumerate(fact_records):
        key = context_id(fact_record, index)
        if key not in contexts:
            raise ValueError(f"facts references unknown context_id {key}")
        facts = fact_record.get("facts")
        if not isinstance(facts, list):
            raise ValueError(f"facts record {key} has no facts list")
        context = contexts[key].get("context")
        if not isinstance(context, str):
            raise ValueError(f"context record {key} has no context string")
        prompts.append(fill_prompt(template, {
            "{{CONTEXT}}": context,
            "{{FACTS_JSON}}": json.dumps(facts, ensure_ascii=False),
        }))
        keys.append(key)

    runner = ModelRunner(args.model, args.device_map, args.torch_dtype)
    raw_outputs = runner.generate(prompts, args.batch_size, args.max_new_tokens)
    qa_records: list[dict[str, Any]] = []
    skipped = 0
    for key, raw in zip(keys, raw_outputs):
        parsed, _ = parse_json_object(raw)
        normalized = normalize_output(parsed)
        if normalized is None:
            skipped += 1
            continue
        qa, evidence = normalized
        qa_records.append({
            "context_id": key, "qa": qa, "evidence": evidence
        })

    qa_output = args.output_dir / "qa.jsonl"
    write_jsonl(qa_output, qa_records)
    print(f"written={len(qa_records)} skipped={skipped} output={qa_output}")
    return qa_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
