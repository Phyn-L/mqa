"""Shared JSONL and model-inference utilities for all stages."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


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


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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


class ModelRunner:
    def __init__(self, model_name: str, device_map: str, torch_dtype: str) -> None:
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
    ) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            messages = [[{"role": "user", "content": prompt}] for prompt in batch]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=True, return_tensors="pt",
                processor_kwargs={"padding": True},
            )
            inputs = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            generation_args: dict[str, Any] = {
                "max_new_tokens": max_new_tokens, "do_sample": sample
            }
            if sample:
                generation_args.update(temperature=0.5, top_p=0.9)
            with torch.inference_mode():
                generated = self.model.generate(**inputs, **generation_args)
            input_width = inputs["input_ids"].shape[-1]
            outputs.extend(
                text.strip()
                for text in self.processor.batch_decode(
                    generated[:, input_width:], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        return outputs
