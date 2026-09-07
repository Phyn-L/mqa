"""Stage 3: validate QA and write only failed context IDs."""
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


DEFAULT_PROMPT = (
    Path(__file__).resolve().parents[1] / "prompts/multiturn_qa_audit_prompt.txt"
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


def merge_record(
    key: str, context_record: dict[str, Any],
    fact_record: dict[str, Any] | None, qa_record: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if fact_record is None or qa_record is None:
        return None
    context = context_record.get("context")
    facts = fact_record.get("facts")
    qa = qa_record.get("qa")
    evidence = qa_record.get("evidence")
    if (
        not isinstance(context, str) or not isinstance(facts, list)
        or not isinstance(qa, dict) or not isinstance(evidence, list)
    ):
        return None
    conversation = qa.get("conversation")
    if not isinstance(conversation, list):
        return None
    evidence_by_turn = {
        item.get("turn_id"): item.get("spans")
        for item in evidence if isinstance(item, dict)
    }
    enriched: list[dict[str, Any]] = []
    for turn in conversation:
        if not isinstance(turn, dict):
            return None
        turn = dict(turn)
        turn["evidence"] = evidence_by_turn.get(turn.get("turn_id"), [])
        enriched.append(turn)
    return {
        "context_id": key, "context": context, "facts": facts,
        "dialogue_plan": qa.get("dialogue_plan"), "conversation": enriched,
    }


def validate_qa(record: dict[str, Any]) -> list[str]:
    context = record["context"]
    facts = record["facts"]
    plan = record.get("dialogue_plan")
    conversation = record.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        return ["conversation must be a non-empty list"]
    errors: list[str] = []
    if not isinstance(plan, list) or len(plan) != len(conversation):
        errors.append("dialogue_plan must contain one item per turn")
    fact_ids = {
        fact["fact_id"] for fact in facts
        if isinstance(fact, dict) and isinstance(fact.get("fact_id"), str)
    }
    previous_turns: set[str] = set()
    seen_turns: set[str] = set()
    for index, turn in enumerate(conversation):
        prefix = f"conversation[{index}]"
        turn_id = turn.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id:
            errors.append(f"{prefix}.turn_id must be non-empty")
        elif turn_id in seen_turns:
            errors.append(f"duplicate turn_id {turn_id}")
        else:
            seen_turns.add(turn_id)
        for field in ("question", "answer", "operation"):
            if not isinstance(turn.get(field), str) or not turn[field].strip():
                errors.append(f"{prefix}.{field} must be non-empty")
        required = turn.get("required_facts")
        if not isinstance(required, list) or not required:
            errors.append(f"{prefix}.required_facts must be a non-empty list")
        elif unknown := [fact_id for fact_id in required if fact_id not in fact_ids]:
            errors.append(f"{prefix} references unknown facts: {unknown}")
        evidence = turn.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{prefix}.evidence must be a non-empty list")
        elif any(not isinstance(span, str) or span not in context for span in evidence):
            errors.append(f"{prefix}.evidence contains a non-context span")
        dependencies = turn.get("depends_on_turns")
        if not isinstance(dependencies, list):
            errors.append(f"{prefix}.depends_on_turns must be a list")
            dependencies = []
        elif any(dependency not in previous_turns for dependency in dependencies):
            errors.append(f"{prefix} depends on a non-previous turn")
        if turn.get("requires_history") is not bool(dependencies):
            errors.append(f"{prefix}.requires_history disagrees with dependencies")
        if isinstance(turn_id, str):
            previous_turns.add(turn_id)
    if len(conversation) < 2:
        errors.append("conversation must contain at least two turns")
    return errors


def build_audit_prompt(template: str, record: dict[str, Any]) -> str:
    dialogue = {
        "dialogue_plan": record["dialogue_plan"],
        "conversation": record["conversation"],
    }
    return fill_prompt(template, {
        "{{CONTEXT_ID}}": record["context_id"],
        "{{CONTEXT}}": record["context"],
        "{{FACTS_JSON}}": json.dumps(record["facts"], ensure_ascii=False),
        "{{DIALOGUE_JSON}}": json.dumps(dialogue, ensure_ascii=False),
    })


def audit_passed(raw: str) -> bool:
    parsed, parse_error = parse_json_object(raw)
    return bool(
        not parse_error and isinstance(parsed, dict)
        and not set(parsed) - {"valid", "reason"}
        and parsed.get("valid") is True
    )


def run(args: argparse.Namespace) -> Path:
    contexts = index_by_context_id(
        read_jsonl(args.contexts, args.max_samples), "contexts"
    )
    facts = index_by_context_id(read_jsonl(args.facts), "facts")
    qa = index_by_context_id(read_jsonl(args.qa), "qa")
    merged: dict[str, dict[str, Any]] = {}
    failed_ids: list[str] = []
    for key, context_record in contexts.items():
        record = merge_record(
            key, context_record, facts.get(key), qa.get(key)
        )
        if record is None or validate_qa(record):
            failed_ids.append(key)
        else:
            merged[key] = record

    if merged:
        template = args.prompt.read_text(encoding="utf-8")
        keys = list(merged)
        prompts = [build_audit_prompt(template, merged[key]) for key in keys]
        runner = ModelRunner(args.model, args.device_map, args.torch_dtype)
        audits = runner.generate(
            prompts, args.batch_size, args.max_new_tokens,
            sortish_window_size=args.sortish_window_size,
            sortish_seed=args.sortish_seed, token_budget=args.token_budget,
        )
        failed_ids.extend(
            key for key, raw in zip(keys, audits) if not audit_passed(raw)
        )

    output = args.output_dir / "rejected.jsonl"
    unique_failed_ids = dict.fromkeys(failed_ids)
    write_jsonl(output, [{"context_id": key} for key in unique_failed_ids])
    print(f"rejected={len(unique_failed_ids)} output={output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--sortish-window-size", type=int, default=2000)
    parser.add_argument("--sortish-seed", type=int, default=42)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
