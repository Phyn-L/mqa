"""Run an existing JSONL inference script once per GPU and merge its outputs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


OUTPUT_FILES = (
    "facts.jsonl", "failed.jsonl", "qa.jsonl", "rejected.jsonl",
    "audit.jsonl", "accepted.jsonl",
)


def load_lines(path: Path, max_samples: int | None) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    return lines[:max_samples] if max_samples is not None else lines


def split_contiguous(lines: list[str], workers: int) -> list[list[str]]:
    """Split into contiguous, disjoint ranges while keeping input order."""
    count, remainder = divmod(len(lines), workers)
    shards: list[list[str]] = []
    start = 0
    for worker in range(workers):
        size = count + (1 if worker < remainder else 0)
        shards.append(lines[start : start + size])
        start += size
    return shards


def _validate_unique_ids(gpus: list[str]) -> None:
    if len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must contain unique GPU ids")


def _without_control_args(args: list[str]) -> list[str]:
    """Remove parent-owned input/output/max-samples flags from child args."""
    result: list[str] = []
    skip_next = False
    owned = {"--input", "--output-dir", "--max-samples"}
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in owned:
            skip_next = True
            continue
        for prefix in ("--input=", "--output-dir=", "--max-samples="):
            if arg.startswith(prefix):
                break
        else:
            result.append(arg)
    return result


def _normalize_source(value: Any, original_input: str) -> Any:
    if isinstance(value, dict):
        value = dict(value)
        if "input_file" in value:
            value["input_file"] = original_input
    return value


def merge_jsonl(
    worker_dirs: list[Path], output_dir: Path, filename: str, original_input: str
) -> int:
    output_path = output_dir / filename
    count = 0
    with output_path.open("w", encoding="utf-8") as destination:
        for worker_dir in worker_dirs:
            path = worker_dir / filename
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if isinstance(record, dict) and "source" in record:
                        record["source"] = _normalize_source(record["source"], original_input)
                    destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
    return count


def merge_summaries(worker_dirs: list[Path], output_dir: Path, total: int) -> None:
    summary: dict[str, Any] = {"total": total, "workers": len(worker_dirs)}
    error_counts: dict[str, int] = {}
    for worker_dir in worker_dirs:
        path = worker_dir / "summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if key == "error_counts" and isinstance(value, dict):
                for error, count in value.items():
                    error_counts[error] = error_counts.get(error, 0) + int(count)
            elif isinstance(value, (int, float)) and key != "total":
                summary[key] = summary.get(key, 0) + value
    if error_counts:
        summary["error_counts"] = error_counts
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run(args: argparse.Namespace, child_args: list[str]) -> None:
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    _validate_unique_ids(gpus)
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    lines = load_lines(args.input, args.max_samples)
    if not lines:
        raise ValueError(f"No records found in {args.input}")
    shards = split_contiguous(lines, len(gpus))
    child_args = _without_control_args(child_args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="parallel_infer_", dir=args.output_dir) as temp_root:
        temp_root_path = Path(temp_root)
        processes: list[tuple[int, subprocess.Popen[str], Path]] = []
        for worker, (gpu, shard) in enumerate(zip(gpus, shards)):
            if not shard:
                continue
            shard_path = temp_root_path / f"input_{worker}.jsonl"
            shard_path.write_text("".join(shard), encoding="utf-8")
            worker_dir = temp_root_path / f"worker_{worker}"
            worker_dir.mkdir()
            command = [
                sys.executable, str(args.script),
                "--input", str(shard_path),
                "--output-dir", str(worker_dir),
                *child_args,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(command, env=env, text=True)
            processes.append((worker, process, worker_dir))

        failures: list[tuple[int, int]] = []
        for worker, process, _ in processes:
            return_code = process.wait()
            if return_code:
                failures.append((worker, return_code))
        if failures:
            details = ", ".join(f"worker {w}: exit {code}" for w, code in failures)
            raise RuntimeError(f"parallel inference failed ({details})")

        worker_dirs = [worker_dir for _, _, worker_dir in processes]
        for filename in OUTPUT_FILES:
            merge_jsonl(worker_dirs, args.output_dir, filename, str(args.input.resolve()))
        merge_summaries(worker_dirs, args.output_dir, len(lines))


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU ids")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("child_args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    child_args = parsed.child_args
    if child_args[:1] == ["--"]:
        child_args = child_args[1:]
    return parsed, child_args


if __name__ == "__main__":
    namespace, remainder = parse_args()
    run(namespace, remainder)
