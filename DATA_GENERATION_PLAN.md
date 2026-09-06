# SHINE Multi-turn QA Data Generation Plan

## 1. Purpose

Build a high-quality multi-turn QA dataset for training and evaluating SHINE LoRA. The data should help the generated LoRA retain important context information across turns, resolve references, combine facts, track updates, synthesize key information, and avoid unsupported claims.

Pipeline: context -> atomic facts -> fact validation -> dialogue plan -> multi-turn QA -> QA validation.

## 2. Existing implementation

The initial extractor is /data/lz/mqa/qa_gen/fact_extraction.py. It reads JSONL contexts, calls a causal language model, parses JSON, validates atomic facts, checks unique IDs, validates text/evidence/importance, verifies evidence substrings, and writes facts.jsonl, failed.jsonl, and summary.json.

The prompt is /data/lz/mqa/qa_gen/fact_extract_prompt.txt and currently extracts only atomic facts. Add fact quality filtering, dialogue planning, multi-turn QA generation, and QA validation. Do not generate dialogues directly from raw contexts without the fact layer.

## 3. Fact representation

Atomic facts contain only information explicitly stated in the context. Each fact has fact_id, text, fact_type, importance, and evidence. Each fact expresses one primary claim, is independently understandable, preserves dates, numbers, units, time, negation, and conditions, and has verbatim contiguous evidence from the context. Atomic facts do not contain derived_from. Extract every explicit fact, including minor details, without external knowledge or duplicates.

Optional derived facts are useful new conclusions requiring at least two atomic facts, such as comparisons, multi-hop relations, or cross-sentence causal relations. A paraphrase of one atomic fact is not derived. Optional summary facts are two to four high-level, traceable facts for final synthesis.

## 4. Generation stages

### A. Prepare contexts

Read contexts from standardized data. Remove empty, corrupted, very short, and exact-duplicate records. Preserve context ID, source dataset, split, and original text. Report context-length and domain distributions.

### B. Extract atomic facts

Run the existing extractor with deterministic or low-temperature structured JSON decoding. Retry failures a limited number of times. Preserve raw output, model name, prompt version, decoding settings, random seed, and code version.

### C. Validate facts

Check JSON format, unique IDs, evidence substring membership, evidence coverage of fact text, independent readability, duplicate facts, compound facts, preservation of numbers/dates/units/negation/time, and sentence-level context coverage. Reject or retry hallucinated, incomplete, unsupported, or incorrectly resolved facts, recording failure reasons.

### D. Plan dialogue dependencies

Create a structured plan before natural-language generation. Each turn specifies turn_id, operation, target fact IDs, dependency turn IDs, and whether a previous answer is required. Operations include retrieve, coreference, ellipsis, comparison, multi_hop, causal_reasoning, update, and synthesis. Typical dialogues have four to eight turns when supported by the context. Do not force unsupported operations.

Each dialogue should normally include a direct retrieval turn, a reference or ellipsis turn, a comparison or multi-hop turn, a causal/consequence turn when supported, and a final synthesis turn.

### E. Generate multi-turn QA

The QA generator receives the original context, validated facts, optional derived/summary facts, and the dialogue plan. Each turn contains turn_id, question, answer, operation, required_facts, depends_on_turns, and evidence.

Later questions must genuinely depend on declared history. If dependency turns are removed, the question should become ambiguous, incomplete, or materially change its answer scope. Use references, inherited entities, inherited values, or constraints from earlier turns. Do not concatenate independent single-turn questions. Answers may use only context, history, and explicit update rules. The final synthesis should cover most high-importance facts.

For update examples, record the original value, update turn, and current value explicitly.

### F. Validate QA

Check every turn schema, required fact IDs, evidence, answer support, numerical and temporal consistency, entity references, and cross-turn state. Run a history-deletion test for every declared dependency. Check final key-fact coverage. Rewrite or reject weak or unsupported dialogues.

## 5. Difficulty, scale, and splits

Pilot distribution: 20% simple contexts with 5–8 facts and 2–3 turns; 40% medium contexts with 8–15 facts and 4–5 turns; 30% difficult contexts with 15–20 facts and comparison or multi-hop reasoning; 10% contexts with updates, conflicts, long context, or complex synthesis.

Start with 100–500 contexts for smoke testing and approximately 5,000 contexts for the first pilot. Split at context level into 80% train, 10% validation, and 10% test. Keep duplicate contexts, paraphrases, and near-identical fact sets in the same split. Where possible, isolate entities, templates, relation combinations, and long-context variants.

## 6. Expected role and deliverables

For training, provide dense supervision for context retrieval, cross-turn entity binding, reference resolution, fact composition, final coverage, and update handling. For research, compare fact-tracked multi-turn QA with independent single-turn QA, and per-turn supervision with final-turn-only supervision. For evaluation, retain fact-level labels for retrieval, reference resolution, multi-hop, update accuracy, summary coverage, old-value suppression, and unsupported-information rate.

Required outputs:

facts/{train,validation,test}.jsonl
qa/{train,validation,test}.jsonl
quality/extraction_report.json
quality/qa_report.json
quality/rejected.jsonl
metadata/generation_config.json
metadata/prompt_versions.json

Every QA record must be traceable through context_id to the source context, facts, dialogue plan, generated QA, and validation result. Record model, prompt, decoding parameters, seed, code version, and timestamp.

## 7. Acceptance criteria and execution order

The pilot is ready for SHINE training only when evidence checks pass, duplicate and compound-fact rates are low, manual sampling shows no systematic omissions or hallucinations, every multi-turn dialogue has required_facts and depends_on_turns, history-deletion tests confirm real dependencies, final turns cover most high-importance facts, leakage is controlled, and failure reasons are measurable.

Execute in this order: run the existing extractor on 100–500 contexts; measure JSON, evidence, duplicate, compound-fact, and missing-fact rates; improve the prompt and validator; freeze the atomic fact schema; implement and validate dialogue plans; generate and validate a small QA pilot; implement history-deletion and cross-turn consistency checks; perform manual sampling; compare extraction models and QA prompts under the same contexts and token budget; expand only after acceptance.

## 8. Non-negotiable constraints

- Fact extraction and QA generation are independently retryable stages.
- The QA generator must not invent facts without fact IDs.
- Programmatic checks must handle JSON, IDs, evidence, numbers, dates, and provenance; an LLM judge alone is insufficient.
- A follow-up-sounding question is not necessarily dependent; require the history-deletion test.
- Do not generate derived or summary facts from unvalidated atomic facts.
- Do not silently keep failed records.
- Preserve provenance from every answer back to the original context.
