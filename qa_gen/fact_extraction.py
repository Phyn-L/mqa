from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor
from tqdm.auto import tqdm


from .utils import (
    PromptItem, configure_generation_padding, decode_generated_tokens,
    load_dataset_records, parse_json_object, read_jsonl, sortish_batches,
)


DEFAULT_PROMPT = Path(__file__).with_name("prompts") / "fact_extract_prompt.txt"
DEFAULT_INPUT = Path(__file__).parent.parent / "standardized" / "squad" / "train-v1.1.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("atomic_fact")

# Prompt construction.

def build_prompt(template: str, context: str) -> str:
    if "{{CONTEXT}}" not in template:
        raise ValueError("Prompt template must contain {{CONTEXT}} placeholder")
    return template.replace("{{CONTEXT}}", context, 1)


# Output validation.

def validate_output(parsed: dict[str, Any] | None, context: str) -> list[str]:
    """Validate model JSON and every fact against the original context.
    Facts and evidence are checked before a record is accepted.
    """
    errors: list[str] = []
    if parsed is None:
        return ["missing parsed object"]
    extra_keys = set(parsed) - {"facts"}
    if extra_keys:
        errors.append(f"unexpected top-level keys: {sorted(extra_keys)}")
    facts = parsed.get("facts")
    if not isinstance(facts, list):
        return ["facts must be a list"]
    seen_ids: set[str] = set()
    for index, fact in enumerate(facts):
        prefix = f"facts[{index}]"
        if not isinstance(fact, dict):
            errors.append(f"{prefix} must be an object")
            continue
        fact_id = fact.get("fact_id")
        if not isinstance(fact_id, str) or not fact_id:
            errors.append(f"{prefix}.fact_id must be a non-empty string")
        elif fact_id in seen_ids:
            errors.append(f"duplicate fact_id: {fact_id}")
        else:
            seen_ids.add(fact_id)
        if not isinstance(fact.get("text"), str) or not fact["text"].strip():
            errors.append(f"{prefix}.text must be a non-empty string")
        evidence = fact.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            errors.append(f"{prefix}.evidence must be a non-empty string")
        elif evidence not in context:
            errors.append(f"{prefix}.evidence is not a contiguous context substring")
        importance = fact.get("importance")
        if not isinstance(importance, (int, float)) or not 0.0 <= float(importance) <= 1.0:
            errors.append(f"{prefix}.importance must be a number in [0, 1]")
    return errors


def canonicalize_output(
    parsed: dict[str, Any], record: dict[str, Any], input_path: Path
) -> dict[str, Any]:
    """Attach trusted provenance and normalize the facts field.

    Model-generated metadata is never trusted: context ID, source and context
    text come from the input record.
    """
    facts = parsed["facts"]
    source = source_metadata(record, input_path)
    source = {key: source[key] for key in ("dataset", "split") if key in source}
    return {
        "context_id": record.get("context_id", record.get("id")),
        "source": source,
        "context": record["context"],
        "facts": facts,
    }


def build_prompt_items(
    records: Sequence[dict[str, Any]],
    template: str,
    processor: Any,
) -> list[PromptItem]:
    """Build prompts and measure their chat-template token lengths.

    Token lengths are computed with the same processor used for generation so
    sortish batching accounts for the prompt wrapper and special tokens.
    """
    # The model only generates facts.  Provenance and the original context are
    # copied from the input record by ``canonicalize_output`` after validation.
    prompts = [build_prompt(template, record["context"]) for record in records]
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(tokenized, dict):
        tokenized = tokenized["input_ids"]
    lengths = [len(ids) for ids in tokenized]
    return [
        PromptItem(index=index, record=record, prompt=prompt, token_length=lengths[index])
        for index, (record, prompt) in enumerate(zip(records, prompts))
    ]


# Model inference.

def generate_batches(
    model: Any,
    processor: Any,
    input_device: torch.device,
    prompt_batches: Sequence[Sequence[PromptItem]],
    max_new_tokens: int,
    *,
    do_sample: bool,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    generation_tokens: int | None = None,
) -> dict[int, str]:
    """Run inference for planned prompt batches and return outputs by index.

    The function receives built prompt batches and returns decoded text keyed
    by input index.
    """
    generated_texts: dict[int, str] = {}
    for prompt_batch in tqdm(prompt_batches, desc="Fact extraction", unit="batch"):
        messages = [[{"role": "user", "content": item.prompt}] for item in prompt_batch]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        inputs = {
            key: value.to(input_device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=generation_tokens or max_new_tokens,
                do_sample=do_sample,
                **(
                    {
                        key: value
                        for key, value in {
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": top_k,
                        }.items()
                        if value is not None
                    }
                    if do_sample
                    else {}
                ),
            )
        decoded = decode_generated_tokens(outputs, inputs, processor)
        generated_texts.update(
            (item.index, text.strip()) for item, text in zip(prompt_batch, decoded)
        )
    return generated_texts


def source_metadata(record: dict[str, Any], input_path: Path) -> dict[str, str]:
    """Keep dataset provenance without copying unrelated source fields."""
    source = {
        key: record[key]
        for key in ("dataset", "split", "source_dataset", "source_split")
        if isinstance(record.get(key), str)
    }
    parts = input_path.resolve().parts
    dataset_root = next((name for name in ("standardized", "aggregated") if name in parts), None)
    if "dataset" not in source and dataset_root is not None:
        dataset_index = parts.index(dataset_root) + 1
        if dataset_index < len(parts):
            source["dataset"] = parts[dataset_index]
    if "split" not in source and input_path.suffix == ".jsonl":
        source["split"] = input_path.stem
    source["input_file"] = str(input_path)
    return source


def validate_generation_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.max_samples is not None and args.max_samples < 1:
        raise ValueError("batch size and max samples must be positive")
    if args.max_attempts < 1 or args.max_new_tokens < 1 or args.retry_max_new_tokens < 1:
        raise ValueError("attempts and generation token limits must be positive")
    if args.retry_temperature <= 0:
        raise ValueError("retry temperature must be greater than 0")
    if not 0 < args.retry_top_p <= 1:
        raise ValueError("retry top-p must be in (0, 1]")
    if args.retry_top_k is not None and args.retry_top_k < 1:
        raise ValueError("retry top-k must be a positive integer")
    if args.token_budget is not None and args.token_budget < 1:
        raise ValueError("token budget must be positive")
    if args.sortish_seed < 0 or args.sortish_window_size < 1:
        raise ValueError("sortish seed must be non-negative")


# Output persistence.

def write_results(
    output_dir: Path,
    final_results: Sequence[dict[str, Any]],
    failed_results: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> Path:
    """Write accepted facts, rejected records, and summary atomically by file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "facts.jsonl"
    failed_path = output_dir / "failed.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for result in final_results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    with failed_path.open("w", encoding="utf-8") as handle:
        for result in failed_results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def run(args: argparse.Namespace) -> Path:
    """Run loading, generation, validation, and output stages."""
    validate_generation_args(args)
    records = load_dataset_records(args.input, args.max_samples)
    if not records:
        raise ValueError(f"No records found in {args.input}")
    processor = AutoProcessor.from_pretrained(args.model)
    configure_generation_padding(processor)
    template = args.prompt.read_text(encoding="utf-8")
    prompt_items = build_prompt_items(records, template, processor)
    batches = sortish_batches(
        prompt_items,
        batch_size=args.batch_size,
        window_size=args.sortish_window_size,
        seed=args.sortish_seed,
        token_budget=args.token_budget,
        # Reserve generation space in the batch token budget.
        generation_tokens=args.max_new_tokens,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, device_map=args.device_map, dtype=args.torch_dtype
    )
    input_device = model.get_input_embeddings().weight.device

    # Use sampling only for retries.
    outputs = generate_batches(
        model, processor, input_device, batches, args.max_new_tokens, do_sample=False
    )
    final_results: list[dict[str, Any]] = []
    failed_results: list[dict[str, Any]] = []
    valid_json = 0
    rejected = 0
    error_counts: dict[str, int] = {}
    for item in tqdm(prompt_items, desc="Validating facts", unit="record"):
        record = item.record
        raw_text = outputs[item.index]
        parsed, parse_error = parse_json_object(raw_text)
        errors = validate_output(parsed, record["context"]) if parse_error is None else [parse_error]
        attempts = 1
        while errors and attempts < args.max_attempts:
            raw_text = generate_batches(
                model,
                processor,
                input_device,
                [[item]],
                args.max_new_tokens,
                generation_tokens=args.retry_max_new_tokens,
                do_sample=args.retry_sample,
                temperature=args.retry_temperature if args.retry_sample else None,
                top_p=args.retry_top_p if args.retry_sample else None,
                top_k=args.retry_top_k if args.retry_sample else None,
            )[item.index]
            parsed, parse_error = parse_json_object(raw_text)
            errors = validate_output(parsed, record["context"]) if parse_error is None else [parse_error]
            attempts += 1
        if parsed is not None:
            valid_json += 1
        if errors:
            rejected += 1
            for error in errors:
                error_counts[error] = error_counts.get(error, 0) + 1
            failed_results.append({
                "id": record.get("id", record.get("context_id")),
                "source": source_metadata(record, args.input),
                "errors": errors,
                "raw_output": raw_text,
                "attempts": attempts,
            })
            continue
        final_results.append(
            canonicalize_output(parsed, record, args.input)
        )

    summary = {
        "total": len(records),
        "valid_json": valid_json,
        "written": len(final_results),
        "rejected": rejected,
        "error_counts": error_counts,
    }
    output_path = write_results(args.output_dir, final_results, failed_results, summary)
    print(json.dumps({"input": str(args.input), "output": str(output_path), "failed": str(args.output_dir / 'failed.jsonl'), **summary}, ensure_ascii=False))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32768,
        help="Maximum generated tokens; Qwen3.5 guidance recommends 32768 for complex queries.",
    )
    parser.add_argument(
        "--retry-max-new-tokens",
        type=int,
        default=32768,
        help="Generation limit for retries after malformed or truncated output.",
    )
    parser.add_argument(
        "--sortish-window-size",
        type=int,
        default=2000,
        help="Number of shuffled examples sorted locally before batching.",
    )
    parser.add_argument("--sortish-seed", type=int, default=42)
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Optional per-batch budget for padded input plus max_new_tokens.",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--retry-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sampling only for retries; the first pass is always deterministic.",
    )
    parser.add_argument("--retry-temperature", type=float, default=0.5)
    parser.add_argument("--retry-top-p", type=float, default=0.9)
    parser.add_argument("--retry-top-k", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
