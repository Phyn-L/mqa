"""Audit conversational QA with deterministic checks and an LLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

try:
    from .utils import (
        PromptItem, configure_generation_padding, decode_generated_tokens,
        load_dataset_records, parse_json_object, sortish_batches,
    )
    from .multiturn_qa_generation import validate_dialogue
    from .multiturn_qa_generation import build_qa_prompt, generate_batches, canonicalize
except ImportError:
    from utils import (
        PromptItem, configure_generation_padding, decode_generated_tokens,
        load_dataset_records, parse_json_object, sortish_batches,
    )
    from multiturn_qa_generation import validate_dialogue
    from multiturn_qa_generation import build_qa_prompt, generate_batches, canonicalize

DEFAULT_PROMPT = Path(__file__).with_name("prompts") / "multiturn_qa_audit_prompt.txt"
DEFAULT_GENERATION_PROMPT = Path(__file__).with_name("prompts") / "multiturn_qa_prompt.txt"
DEFAULT_INPUT = Path(__file__).with_name("generated_multiturn_qa") / "qa.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("multiturn_qa_audit")


def load_qa_records(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    return load_dataset_records(path, max_samples)


def build_audit_prompt(template: str, record: dict[str, Any]) -> str:
    replacements = {
        "{{CONTEXT_ID}}": str(record.get("context_id", record.get("id", ""))),
        "{{CONTEXT}}": record.get("context", ""),
        "{{FACTS_JSON}}": json.dumps(record.get("facts", []), ensure_ascii=False),
        "{{DIALOGUE_JSON}}": json.dumps({"dialogue_plan": record.get("dialogue_plan", []), "conversation": record.get("conversation", [])}, ensure_ascii=False),
    }
    for key, value in replacements.items():
        if key not in template:
            raise ValueError(f"audit prompt is missing {key}")
        template = template.replace(key, value)
    return template


def build_prompt_items(records: list[dict[str, Any]], template: str, processor: Any) -> list[PromptItem]:
    prompts = [build_audit_prompt(template, record) for record in records]
    messages = [[{"role": "user", "content": p}] for p in prompts]
    tokenized = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized
    return [PromptItem(i, r, prompts[i], len(ids[i])) for i, r in enumerate(records)]


def generate_audits(model: Any, processor: Any, device: torch.device, batches: list[list[PromptItem]], max_new_tokens: int) -> dict[int, str]:
    outputs: dict[int, str] = {}
    for batch in batches:
        messages = [[{"role": "user", "content": x.prompt}] for x in batch]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors="pt", processor_kwargs={"padding": True})
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        outputs.update({x.index: t.strip() for x, t in zip(batch, decode_generated_tokens(generated, inputs, processor))})
    return outputs


def audit_record(record: dict[str, Any], raw_audit: str) -> dict[str, Any]:
    deterministic_errors = validate_dialogue(record, record)
    parsed, parse_error = parse_json_object(raw_audit)
    llm_errors: list[str] = []
    if parse_error:
        llm_errors.append(parse_error)
    elif not isinstance(parsed.get("valid"), bool):
        llm_errors.append("audit JSON must contain boolean valid")
    elif parsed.get("valid") is False and (
        not isinstance(parsed.get("reason"), str) or not parsed["reason"].strip()
    ):
        llm_errors.append("invalid audit result must include a non-empty reason")
    llm_pass = bool(parsed and parsed.get("valid") is True and not llm_errors)
    return {
        "context_id": record.get("context_id"),
        "deterministic_errors": deterministic_errors,
        "llm_pass": llm_pass,
        "llm_errors": llm_errors,
        "llm_audit": parsed,
        "accepted": not deterministic_errors and llm_pass,
    }


def build_regeneration_prompt(
    generation_template: str, record: dict[str, Any], failure_reasons: list[str],
    previous_dialogue: dict[str, Any] | None = None,
) -> str:
    """Create a repair prompt while preserving the original source contract."""
    prompt = build_qa_prompt(generation_template, record)
    previous = json.dumps(previous_dialogue or {}, ensure_ascii=False)
    reasons = json.dumps(failure_reasons, ensure_ascii=False)
    return (
        prompt
        + "\n\nA previous candidate failed quality control. Regenerate the COMPLETE JSON object; do not output a patch or commentary."
        + "\nAudit failure reasons (fix every item):\n" + reasons
        + "\nPrevious candidate dialogue (use only to identify and repair errors):\n" + previous
    )


def audit_with_regeneration(
    model: Any, processor: Any, device: torch.device, record: dict[str, Any],
    audit_template: str, generation_template: str, max_new_tokens: int,
    max_regenerations: int, first_result: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Audit a candidate and regenerate it after failures up to a fixed limit."""
    attempts: list[dict[str, Any]] = []
    candidate = record
    for attempt in range(max_regenerations + 1):
        if attempt == 0 and first_result is not None:
            result = dict(first_result)
        else:
            audit_prompt = build_audit_prompt(audit_template, candidate)
            item = PromptItem(0, candidate, audit_prompt, 0)
            raw_audit = generate_audits(model, processor, device, [[item]], max_new_tokens)[0]
            result = audit_record(candidate, raw_audit)
        result["attempt"] = attempt
        attempts.append(result)
        if result["accepted"]:
            result["attempts"] = [dict(item) for item in attempts]
            return candidate, result
        if attempt >= max_regenerations:
            break
        reasons = list(result.get("deterministic_errors", []))
        reasons.extend(result.get("llm_errors", []))
        llm = result.get("llm_audit") or {}
        if isinstance(llm, dict) and isinstance(llm.get("reason"), str):
            reasons.append(llm["reason"])
        repair_prompt = build_regeneration_prompt(
            generation_template, record, reasons,
            {"dialogue_plan": candidate.get("dialogue_plan", []), "conversation": candidate.get("conversation", [])},
        )
        repair_item = PromptItem(0, record, repair_prompt, 0)
        raw_candidate = generate_batches(
            model, processor, device, [[repair_item]], max_new_tokens, sample=True
        )[0]
        parsed, parse_error = parse_json_object(raw_candidate)
        if parse_error or parsed is None:
            # Do not retain the previous valid conversation after malformed
            # regeneration output; otherwise it could be audited as if it
            # were the newly generated candidate.
            candidate = {
                "context_id": record.get("context_id"),
                "source": record.get("source", {}),
                "context": record.get("context", ""),
                "facts": record.get("facts", []),
                "dialogue_plan": [],
                "conversation": [],
                "_regeneration_parse_error": parse_error or "invalid JSON",
            }
        else:
            # Keep source provenance and facts trusted even on a repaired output.
            candidate = canonicalize(parsed, record)
    final = attempts[-1]
    final["attempts"] = [dict(item) for item in attempts]
    return None, final


def run(args: argparse.Namespace) -> Path:
    if args.max_regenerations < 0:
        raise ValueError("max-regenerations must be non-negative")
    records = load_qa_records(args.input, args.max_samples)
    template = args.prompt.read_text(encoding="utf-8")
    generation_template = args.generation_prompt.read_text(encoding="utf-8")
    processor = AutoProcessor.from_pretrained(args.model); configure_generation_padding(processor)
    items = build_prompt_items(records, template, processor)
    batches = sortish_batches(items, args.batch_size, args.sortish_window_size, args.sortish_seed, args.token_budget, args.max_new_tokens)
    model = AutoModelForMultimodalLM.from_pretrained(args.model, device_map=args.device_map, dtype=args.torch_dtype)
    device = model.get_input_embeddings().weight.device
    raw = generate_audits(model, processor, device, batches, args.max_new_tokens)
    audits: list[dict[str, Any]] = []
    accepted_records: dict[str, dict[str, Any]] = {}
    for item in items:
        initial = audit_record(item.record, raw[item.index])
        if initial["accepted"] or args.max_regenerations == 0:
            initial["attempts"] = [dict(initial)]
            audits.append(initial)
            if initial["accepted"]:
                accepted_records[str(item.record.get("context_id"))] = item.record
            continue
        candidate, final = audit_with_regeneration(
            model, processor, device, item.record, template, generation_template,
            args.max_new_tokens, args.max_regenerations, first_result=initial,
        )
        audits.append(final)
        if candidate is not None and final["accepted"]:
            accepted_records[str(item.record.get("context_id"))] = candidate
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as h:
        h.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in audits)
    accepted_ids = {x["context_id"] for x in audits if x["accepted"]}
    with (args.output_dir / "accepted.jsonl").open("w", encoding="utf-8") as h:
        h.writelines(json.dumps(accepted_records[str(r.get("context_id"))], ensure_ascii=False) + "\n" for r in records if r.get("context_id") in accepted_ids)
    with (args.output_dir / "rejected.jsonl").open("w", encoding="utf-8") as h:
        h.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records if r.get("context_id") not in accepted_ids)
    regenerated = sum(1 for x in audits if len(x.get("attempts", [])) > 1)
    summary = {"total": len(records), "accepted": len(accepted_ids), "rejected": len(records)-len(accepted_ids), "regenerated": regenerated, "max_regenerations": args.max_regenerations}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"audit": str(audit_path), **summary}, ensure_ascii=False))
    return audit_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT); p.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT); p.add_argument("--generation-prompt", type=Path, default=DEFAULT_GENERATION_PROMPT); p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR); p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--max-samples", type=int, default=2); p.add_argument("--batch-size", type=int, default=1); p.add_argument("--max-new-tokens", type=int, default=8192); p.add_argument("--max-regenerations", type=int, default=2, help="Maximum repair generations after the initial failed audit."); p.add_argument("--sortish-window-size", type=int, default=2000); p.add_argument("--sortish-seed", type=int, default=42); p.add_argument("--token-budget", type=int, default=None); p.add_argument("--device-map", default="auto"); p.add_argument("--torch-dtype", default="auto")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
