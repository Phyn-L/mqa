"""Shared JSONL and model-inference utilities for all stages."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Record at {path}:{line_number} must be an object")
            records.append(record)
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def context_id(record: dict[str, Any], index: int) -> str:
    value = record.get("context_id", record.get("id"))
    if value is None or not str(value).strip():
        raise ValueError(f"Record {index} has no context_id or id")
    return str(value)


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.I | re.S)
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


def fill_prompt(template: str, values: dict[str, str]) -> str:
    for placeholder, value in values.items():
        if placeholder not in template:
            raise ValueError(f"Prompt is missing placeholder {placeholder}")
        template = template.replace(placeholder, value)
    return template


def sortish_batches(
    lengths: list[int],
    batch_size: int,
    window_size: int,
    seed: int,
    token_budget: int | None,
    generation_tokens: int,
) -> list[list[int]]:
    """Locally sort prompt indices and pack batches under a token budget."""
    if batch_size < 1 or window_size < 1:
        raise ValueError("batch_size and window_size must be positive")
    if token_budget is not None and token_budget < 1:
        raise ValueError("token_budget must be positive")
    order = list(range(len(lengths)))
    random.Random(seed).shuffle(order)
    locally_sorted: list[int] = []
    for start in range(0, len(order), window_size):
        window = order[start : start + window_size]
        locally_sorted.extend(sorted(window, key=lambda index: lengths[index]))

    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in locally_sorted:
        candidate_max = max(current_max, lengths[index])
        exceeds_budget = (
            token_budget is not None
            and current
            and (candidate_max + generation_tokens) * (len(current) + 1) > token_budget
        )
        if current and (len(current) >= batch_size or exceeds_budget):
            batches.append(current)
            current, current_max = [], 0
        current.append(index)
        current_max = max(current_max, lengths[index])
    if current:
        batches.append(current)
    return batches


class ModelRunner:
    def __init__(self, model_name: str, device_map: str, torch_dtype: str) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name, device_map=device_map, dtype=torch_dtype
        )
        self.device = self.model.get_input_embeddings().weight.device

    def generate(
        self,
        prompts: list[str],
        batch_size: int,
        max_new_tokens: int,
        *,
        sample: bool = False,
        sortish_window_size: int = 2000,
        sortish_seed: int = 42,
        token_budget: int | None = None,
        progress_desc: str | None = None,
    ) -> list[str]:
        if not prompts:
            return []

        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        tokenized = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized
        lengths = [len(ids) for ids in input_ids]
        batches = sortish_batches(
            lengths, batch_size, sortish_window_size, sortish_seed,
            token_budget, max_new_tokens,
        )

        batch_iterator: Iterable[list[int]] = batches
        if progress_desc is not None:
            try:
                from tqdm.auto import tqdm

                batch_iterator = tqdm(
                    batches, total=len(batches), desc=progress_desc, unit="batch"
                )
            except ImportError:
                pass

        outputs: list[str | None] = [None] * len(prompts)
        for batch_indices in batch_iterator:
            batch = [prompts[index] for index in batch_indices]
            messages = [[{"role": "user", "content": prompt}] for prompt in batch]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=True, return_tensors="pt",
                processor_kwargs={"padding": True},
            )
            inputs = {
                key: value.to(self.device)
                if isinstance(value, self.torch.Tensor) else value
                for key, value in inputs.items()
            }
            generation_args: dict[str, Any] = {
                "max_new_tokens": max_new_tokens, "do_sample": sample
            }
            if sample:
                generation_args.update(temperature=0.5, top_p=0.9)
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, **generation_args)
            input_width = inputs["input_ids"].shape[-1]
            decoded = self.processor.batch_decode(
                generated[:, input_width:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for index, text in zip(batch_indices, decoded):
                outputs[index] = text.strip()
        return [text or "" for text in outputs]
