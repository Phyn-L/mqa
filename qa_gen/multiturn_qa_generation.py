"""Generate conversational multi-turn QA from contexts and atomic facts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - keeps the script usable without tqdm
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from .utils import (
        PromptItem, configure_generation_padding, decode_generated_tokens,
        load_dataset_records, parse_json_object, sortish_batches,
    )
except ImportError:
    from utils import (
        PromptItem, configure_generation_padding, decode_generated_tokens,
        load_dataset_records, parse_json_object, sortish_batches,
    )

DEFAULT_PROMPT = Path(__file__).with_name("prompts") / "multiturn_qa_prompt.txt"
DEFAULT_INPUT = Path(__file__).with_name("generated_fact_extraction_small") / "facts.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("generated_multiturn_qa")


def load_fact_records(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load extracted-fact JSONL and enforce the minimum source contract."""
    records = load_dataset_records(path, max_samples)
    for index, record in enumerate(records):
        facts = record.get("facts", record.get("atomic_facts"))
        if not isinstance(facts, list):
            raise ValueError(f"record {index} has no facts list")
        record["facts"] = facts
    return records


def build_qa_prompt(template: str, record: dict[str, Any]) -> str:
    """Fill all prompt placeholders using trusted input metadata."""
    replacements = {
        "{{CONTEXT_ID}}": str(record.get("context_id", record.get("id", ""))),
        "{{SOURCE_JSON}}": json.dumps(record.get("source", {}), ensure_ascii=False),
        "{{FACTS_JSON}}": json.dumps(record["facts"], ensure_ascii=False),
        "{{CONTEXT}}": record["context"],
    }
    missing = [key for key in replacements if key not in template]
    if missing:
        raise ValueError(f"QA prompt is missing placeholders: {', '.join(missing)}")
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def build_prompt_items(records: Sequence[dict[str, Any]], template: str, processor: Any) -> list[PromptItem]:
    """Construct prompts and measure chat-template token lengths for bucketing."""
    prompts = [build_qa_prompt(template, record) for record in records]
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    tokenized = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized
    return [PromptItem(i, record, prompts[i], len(ids[i])) for i, record in enumerate(records)]


def generate_batches(model: Any, processor: Any, device: torch.device, batches: Sequence[Sequence[PromptItem]], max_new_tokens: int, sample: bool = False) -> dict[int, str]:
    """Generate one output per prompt batch; no parsing or file I/O occurs here."""
    outputs: dict[int, str] = {}
    for batch in tqdm(batches, desc="QA generation", unit="batch"):
        messages = [[{"role": "user", "content": item.prompt}] for item in batch]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors="pt", processor_kwargs={"padding": True},
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=sample, **({"temperature": 0.6, "top_p": 0.9} if sample else {}))
        texts = decode_generated_tokens(generated, inputs, processor)
        outputs.update({item.index: text.strip() for item, text in zip(batch, texts)})
    return outputs


def _fact_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(f.get("fact_id")): f for f in record["facts"] if isinstance(f, dict) and f.get("fact_id")}


def validate_dialogue(parsed: dict[str, Any] | None, record: dict[str, Any]) -> list[str]:
    """Deterministically check schema, provenance, evidence, references, and ordering."""
    if not isinstance(parsed, dict):
        return ["output is not a JSON object"]
    errors: list[str] = []
    if parsed.get("context_id") != record.get("context_id", record.get("id")):
        errors.append("context_id does not match input")
    if parsed.get("context") != record.get("context"):
        errors.append("context was modified")
    if parsed.get("source") != record.get("source", {}):
        errors.append("source metadata was modified")
    if parsed.get("facts") != record.get("facts"):
        errors.append("facts were modified")
    conversation = parsed.get("conversation")
    plan = parsed.get("dialogue_plan")
    if not isinstance(conversation, list) or not conversation:
        return errors + ["conversation must be a non-empty list"]
    if not isinstance(plan, list) or len(plan) != len(conversation):
        errors.append("dialogue_plan must have one entry per turn")
    fmap = _fact_map(record)
    turn_ids: list[str] = []
    previous: set[str] = set()
    covered: set[str] = set()
    for i, turn in enumerate(conversation):
        prefix = f"conversation[{i}]"
        if not isinstance(turn, dict):
            errors.append(f"{prefix} must be an object"); continue
        tid = turn.get("turn_id")
        if not isinstance(tid, str) or not tid:
            errors.append(f"{prefix}.turn_id is required"); continue
        if tid in turn_ids:
            errors.append(f"duplicate turn_id: {tid}")
        turn_ids.append(tid)
        for field in ("question", "answer"):
            if not isinstance(turn.get(field), str) or not turn[field].strip():
                errors.append(f"{prefix}.{field} must be non-empty")
        required = turn.get("required_facts")
        if not isinstance(required, list) or not required or not all(isinstance(x, str) for x in required):
            errors.append(f"{prefix}.required_facts must be a non-empty list")
        else:
            unknown = [x for x in required if x not in fmap]
            if unknown: errors.append(f"{prefix} unknown required_facts: {unknown}")
            covered.update(x for x in required if x in fmap)
        evidence = turn.get("evidence")
        if not isinstance(evidence, list) or not evidence or not all(isinstance(x, str) and x for x in evidence):
            errors.append(f"{prefix}.evidence must be a non-empty list of strings")
        else:
            for span in evidence:
                if span not in record["context"]:
                    errors.append(f"{prefix}.evidence is not a context substring"); break
        deps = turn.get("depends_on_turns", [])
        if not isinstance(deps, list) or not all(isinstance(x, str) for x in deps):
            errors.append(f"{prefix}.depends_on_turns must be a list of strings"); deps = []
        if any(dep not in previous for dep in deps):
            errors.append(f"{prefix} depends_on_turns must point to earlier turns")
        if i == 0 and (deps or turn.get("requires_history") is True):
            errors.append("first turn cannot require history")
        if bool(turn.get("requires_history")) != bool(deps):
            errors.append(f"{prefix} requires_history disagrees with depends_on_turns")
        if deps and not any(marker in turn.get("question", "").lower() for marker in (
            "it", "they", "them", "this", "that", "these", "those",
            "he", "she", "which", "what happened", "how did", "why did",
            "compared", "earlier", "previous", "next",
        )):
            errors.append(f"{prefix} declared history dependency lacks an anaphora/ellipsis cue")
        previous.add(tid)
    if len(turn_ids) < 2:
        errors.append("conversation must contain at least two turns")
    if len(fmap) and len(covered) / len(fmap) < 0.5:
        errors.append("conversation covers fewer than half of atomic facts")
    return errors


def canonicalize(parsed: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Trust input provenance/facts while retaining model dialogue fields."""
    return {
        "context_id": record.get("context_id", record.get("id")),
        "source": record.get("source", {}), "context": record["context"],
        "facts": record["facts"], "dialogue_plan": parsed.get("dialogue_plan", []),
        "conversation": parsed["conversation"],
    }


def run(args: argparse.Namespace) -> Path:
    records = load_fact_records(args.input, args.max_samples)
    template = args.prompt.read_text(encoding="utf-8")
    processor = AutoProcessor.from_pretrained(args.model)
    configure_generation_padding(processor)
    items = build_prompt_items(records, template, processor)
    batches = sortish_batches(items, args.batch_size, args.sortish_window_size, args.sortish_seed, args.token_budget, args.max_new_tokens)
    model = AutoModelForMultimodalLM.from_pretrained(args.model, device_map=args.device_map, dtype=args.torch_dtype)
    device = model.get_input_embeddings().weight.device
    raw_outputs = generate_batches(model, processor, device, batches, args.max_new_tokens)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in tqdm(items, desc="Validating QA", unit="record"):
        raw = raw_outputs[item.index]; parsed, parse_error = parse_json_object(raw)
        errors = [parse_error] if parse_error else validate_dialogue(parsed, item.record)
        if errors:
            rejected.append({"context_id": item.record.get("context_id"), "errors": errors, "raw_output": raw})
        else:
            accepted.append(canonicalize(parsed, item.record))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "qa.jsonl"
    with out.open("w", encoding="utf-8") as h:
        h.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in accepted)
    with (args.output_dir / "rejected.jsonl").open("w", encoding="utf-8") as h:
        h.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rejected)
    (args.output_dir / "summary.json").write_text(json.dumps({"total": len(records), "accepted": len(accepted), "rejected": len(rejected)}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "accepted": len(accepted), "rejected": len(rejected)}, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT); p.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR); p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--max-samples", type=int, default=None); p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=32768); p.add_argument("--sortish-window-size", type=int, default=2000); p.add_argument("--sortish-seed", type=int, default=42)
    p.add_argument("--token-budget", type=int, default=None); p.add_argument("--device-map", default="auto"); p.add_argument("--torch-dtype", default="auto")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
