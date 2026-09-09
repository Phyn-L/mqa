"""Stage 0b: reorganize completed spaCy candidates with an LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from tqdm import tqdm

from spacy_utils import compact_parser_candidates
from utils import (
    IncrementalJsonlWriter,
    ModelRunner,
    context_id,
    fill_prompt,
    read_jsonl,
)

DEFAULT_PROMPT = (
    Path(__file__).with_name("prompts") / "candidate_reorganization_prompt.txt"
)
SECTION_HEADERS = {
    "Events:",
    "Participants and support:",
    "Quantities and comparisons:",
    "Relations:",
    "Priority sentences:",
}


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


def validate_candidate_text(raw: str) -> tuple[str, list[str]]:
    text = raw.strip()
    if not text:
        return text, ["model output is empty"]
    errors: list[str] = []
    lines = [line.strip() for line in text.splitlines()]
    if lines[0] != "High-priority candidate information:":
        errors.append("missing High-priority candidate information header")
    if not any(line in SECTION_HEADERS for line in lines):
        errors.append("missing candidate section")
    if not any(line.startswith("- ") and len(line) > 2 for line in lines):
        errors.append("missing candidate bullet")
    if "```" in text or text.startswith("{"):
        errors.append("output must be plain text, not JSON or a code fence")
    return text, errors


def run(args: argparse.Namespace) -> Path:
    contexts_path = args.contexts_dir / args.dataset / "contexts.jsonl"
    contexts = index_by_context_id(read_jsonl(contexts_path), "contexts")
    spacy_candidates_path = (
        args.spacy_candidates_dir / args.dataset / "spacy_candidates.jsonl"
    )
    spacy_records = read_jsonl(spacy_candidates_path, args.max_samples)
    output = args.output_dir / args.dataset / "processed_candidates.jsonl"
    failed_output = args.output_dir / args.dataset / "failed.jsonl"

    # only unprocessed records will pass
    completed = {}
    if output.exists():
        for index, record in enumerate(read_jsonl(output)):
            key = context_id(record, index)
            if (
                isinstance(record.get("candidates"), str)
                and record["candidates"].strip()
            ):
                if key in completed:
                    raise ValueError(
                        f"processed candidates contains duplicate context_id {key}"
                    )
                completed[key] = record
    print(f"{len(completed)}/{len(spacy_records)} records has been processed")

    ids: list[str] = []
    texts: list[str] = []
    seeds: list[dict[str, Any]] = []
    for index, record in tqdm(
        enumerate(spacy_records),
        total=len(spacy_records),
        desc="collecting spaCy records",
    ):
        key = context_id(record, index)
        if key in completed:
            continue
        if key not in contexts:
            raise ValueError(f"spaCy candidates references unknown context_id {key}")
        text = contexts[key].get("context")
        candidate_payload = record.get("candidates")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Context {key} has no non-empty context")
        if not isinstance(candidate_payload, dict):
            raise ValueError(f"spaCy record {key} has no candidates object")
        ids.append(key)
        texts.append(text)
        seeds.append(candidate_payload)

    template = args.prompt.read_text(encoding="utf-8")
    prompts = [
        fill_prompt(
            template,
            {
                "{{CONTEXT}}": text,
                "{{PARSER_CANDIDATES_JSON}}": json.dumps(
                    compact_parser_candidates(candidate),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        )
        for text, candidate in zip(texts, seeds)
    ]
    writer = IncrementalJsonlWriter(output, failed_output)
    with writer:
        if prompts:
            runner = ModelRunner(args.model, args.device_map, args.torch_dtype)

            def write_result(index: int, raw: str) -> None:
                key = ids[index]
                parsed, errors = validate_candidate_text(raw)
                if errors:
                    writer.write_failure(key, raw)
                else:
                    writer.write_success({"context_id": key, "candidates": parsed})

            # Sortish batching may generate contexts out of input order. The
            # callback writes each result as soon as its batch finishes, while
            # context_id keeps the JSONL order-independent and resumable.

            sampling_args = {
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
            runner.generate(
                prompts,
                args.batch_size,
                args.max_new_tokens,
                **sampling_args,
                sortish_window_size=args.sortish_window_size,
                sortish_seed=args.sortish_seed,
                token_budget=args.token_budget,
                progress_desc="LLM reorganization",
                result_callback=write_result,
            )

    print(
        f"existing={len(completed)} written={writer.written} failed={writer.failed} "
        f"output={output} failed_output={failed_output}"
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--contexts-dir", type=Path, default=Path("/data/lz/contexts/aggregated")
    )
    parser.add_argument(
        "--spacy-candidates-dir",
        type=Path,
        default=Path("/data/lz/contexts/spacy_candidates"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/lz/contexts/processed_candidates"),
    )
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=3)

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    # Kept for CLI compatibility with older launch scripts; text output is not retried.
    parser.add_argument("--sortish-window-size", type=int, default=2048)
    parser.add_argument("--sortish-seed", type=int, default=42)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
