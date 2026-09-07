#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL="${MODEL:-/home/lz/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
OUTPUT_ROOT="${OUTPUT_ROOT:-qa_gen/nightly_squad_coqa}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
FACT_MAX_NEW_TOKENS="${FACT_MAX_NEW_TOKENS:-32768}"
FACT_RETRY_MAX_NEW_TOKENS="${FACT_RETRY_MAX_NEW_TOKENS:-32768}"
FACT_MAX_ATTEMPTS="${FACT_MAX_ATTEMPTS:-3}"
QA_MAX_NEW_TOKENS="${QA_MAX_NEW_TOKENS:-32768}"
AUDIT_MAX_NEW_TOKENS="${AUDIT_MAX_NEW_TOKENS:-8192}"
MAX_REGENERATIONS="${MAX_REGENERATIONS:-2}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1048576}"
FORCE_RERUN="${FORCE_RERUN:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/lz/hf_cache}"
export HF_HOME="${HF_HOME:-/home/lz/hf_cache}"
export PYTHONUNBUFFERED=1

run_python() { conda run -n shine python "$@"; }
run_parallel() {
  # The launcher shards the input into disjoint contiguous ranges, binds one
  # child process to each GPU, and merges worker JSONL files in shard order.
  conda run -n shine python qa_gen/parallel_infer.py --gpus "$GPU_IDS" "$@"
}

run_dataset() {
  local dataset="$1" input_path="$2"
  local out_dir="$OUTPUT_ROOT/$dataset"
  local log_path="$out_dir/pipeline.log"
  mkdir -p "$out_dir"
  exec > >(tee -a "$log_path") 2>&1
  echo "[$(date -Is)] start dataset=$dataset input=$input_path"
  echo "[$(date -Is)] model=$MODEL gpus=$CUDA_VISIBLE_DEVICES max_samples=${MAX_SAMPLES:-all}"

  local fact_dir="$out_dir/facts" qa_dir="$out_dir/qa" audit_dir="$out_dir/audit"
  if [[ "$FORCE_RERUN" == 1 || ! -f "$fact_dir/summary.json" ]]; then
    local args=(qa_gen/fact_extraction.py --input "$input_path" --prompt qa_gen/prompts/fact_extract_prompt.txt --output-dir "$fact_dir" --model "$MODEL" --batch-size "$BATCH_SIZE" --max-new-tokens "$FACT_MAX_NEW_TOKENS" --retry-max-new-tokens "$FACT_RETRY_MAX_NEW_TOKENS" --max-attempts "$FACT_MAX_ATTEMPTS" --device-map auto --torch-dtype auto)
    [[ -n "$MAX_SAMPLES" ]] && args+=(--max-samples "$MAX_SAMPLES")
    [[ -n "$TOKEN_BUDGET" ]] && args+=(--token-budget "$TOKEN_BUDGET")
    local parallel_args=(--script qa_gen/fact_extraction.py --input "$input_path" --output-dir "$fact_dir")
    [[ -n "$MAX_SAMPLES" ]] && parallel_args+=(--max-samples "$MAX_SAMPLES")
    echo "[$(date -Is)] fact extraction"; run_parallel "${parallel_args[@]}" -- "${args[@]:1}"
  else echo "[$(date -Is)] skip fact extraction (summary exists)"; fi

  if [[ "$FORCE_RERUN" == 1 || ! -f "$qa_dir/summary.json" ]]; then
    local args=(qa_gen/multiturn_qa_generation.py --input "$fact_dir/facts.jsonl" --prompt qa_gen/prompts/multiturn_qa_prompt.txt --output-dir "$qa_dir" --model "$MODEL" --batch-size "$BATCH_SIZE" --max-new-tokens "$QA_MAX_NEW_TOKENS" --device-map auto --torch-dtype auto)
    [[ -n "$MAX_SAMPLES" ]] && args+=(--max-samples "$MAX_SAMPLES")
    [[ -n "$TOKEN_BUDGET" ]] && args+=(--token-budget "$TOKEN_BUDGET")
    local parallel_args=(--script qa_gen/multiturn_qa_generation.py --input "$fact_dir/facts.jsonl" --output-dir "$qa_dir")
    [[ -n "$MAX_SAMPLES" ]] && parallel_args+=(--max-samples "$MAX_SAMPLES")
    echo "[$(date -Is)] multi-turn QA generation"; run_parallel "${parallel_args[@]}" -- "${args[@]:1}"
  else echo "[$(date -Is)] skip QA generation (summary exists)"; fi

  if [[ "$FORCE_RERUN" == 1 || ! -f "$audit_dir/summary.json" ]]; then
    local args=(qa_gen/multiturn_qa_audit.py --input "$qa_dir/qa.jsonl" --prompt qa_gen/prompts/multiturn_qa_audit_prompt.txt --generation-prompt qa_gen/prompts/multiturn_qa_prompt.txt --output-dir "$audit_dir" --model "$MODEL" --batch-size "$BATCH_SIZE" --max-new-tokens "$AUDIT_MAX_NEW_TOKENS" --max-regenerations "$MAX_REGENERATIONS" --device-map auto --torch-dtype auto)
    [[ -n "$MAX_SAMPLES" ]] && args+=(--max-samples "$MAX_SAMPLES")
    [[ -n "$TOKEN_BUDGET" ]] && args+=(--token-budget "$TOKEN_BUDGET")
    local parallel_args=(--script qa_gen/multiturn_qa_audit.py --input "$qa_dir/qa.jsonl" --output-dir "$audit_dir")
    [[ -n "$MAX_SAMPLES" ]] && parallel_args+=(--max-samples "$MAX_SAMPLES")
    echo "[$(date -Is)] LLM audit and regeneration"; run_parallel "${parallel_args[@]}" -- "${args[@]:1}"
  else echo "[$(date -Is)] skip audit (summary exists)"; fi
  echo "[$(date -Is)] completed dataset=$dataset"
}

run_dataset squad aggregated/squad/train.jsonl
run_dataset coqa aggregated/coqa/train.jsonl
echo "[$(date -Is)] all datasets completed; outputs=$OUTPUT_ROOT"
