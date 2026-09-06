"""Shared data, batching, and decoder-only generation utilities."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


def load_dataset_records(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load JSONL records and validate the minimal context contract."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("context"), str):
                raise ValueError(f"Each record must be an object with a string 'context' (line {line_number})")
            records.append(record)
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


read_jsonl = load_dataset_records


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse strict JSON, tolerating a markdown fence or surrounding text."""
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        candidates.insert(0, fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, None
    return None, "model output is not a JSON object"


@dataclass(frozen=True)
class PromptItem:
    """A prompt and its source record, kept together through batching."""

    index: int
    record: dict[str, Any]
    prompt: str
    token_length: int


def sortish_batches(
    items: Sequence[PromptItem], batch_size: int, window_size: int, seed: int,
    token_budget: int | None = None, generation_tokens: int = 0,
) -> list[list[PromptItem]]:
    """Shuffle, locally sort by length, then pack bounded batches."""
    if batch_size < 1 or window_size < 1:
        raise ValueError("batch_size and window_size must be positive")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    ordered: list[PromptItem] = []
    for start in range(0, len(shuffled), window_size):
        window = shuffled[start : start + window_size]
        ordered.extend(sorted(window, key=lambda item: item.token_length))
    batches: list[list[PromptItem]] = []
    current: list[PromptItem] = []
    current_max = 0
    for item in ordered:
        candidate_max = max(current_max, item.token_length)
        exceeds_budget = (
            token_budget is not None
            and current
            and (candidate_max + generation_tokens) * (len(current) + 1) > token_budget
        )
        if current and (len(current) >= batch_size or exceeds_budget):
            batches.append(current)
            current, current_max = [], 0
        current.append(item)
        current_max = max(current_max, item.token_length)
    if current:
        batches.append(current)
    return batches


def configure_generation_padding(processor: Any) -> None:
    """Configure left padding required by decoder-only generation."""
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            tokenizer.pad_token_id = eos_token_id


def decode_generated_tokens(outputs: torch.Tensor, inputs: dict[str, Any], processor: Any) -> list[str]:
    """Decode only tokens after the shared padded prompt width."""
    input_width = inputs["input_ids"].shape[-1]
    return processor.batch_decode(
        outputs[:, input_width:], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
