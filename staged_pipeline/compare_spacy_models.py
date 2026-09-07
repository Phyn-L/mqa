"""Compare installed spaCy pipelines and emit compact candidate information."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import spacy


CAUSAL = {"because", "therefore", "due", "causing", "escalating"}
COMPARISON = {"compared", "than", "more", "less", "outnumbered"}
TEMPORAL = {"before", "after", "during", "while", "start", "later", "in"}
NEGATION = {"not", "no", "never", "without", "neither", "nor"}


def sentence_reason(sent: Any) -> list[str]:
    lemmas = {token.lemma_.lower() for token in sent}
    lower = sent.text.lower()
    reasons: set[str] = set()
    if sent.ents:
        reasons.add("named_entities")
    if any(token.like_num or token.ent_type_ in {"DATE", "TIME", "CARDINAL", "QUANTITY", "PERCENT", "MONEY"} for token in sent):
        reasons.add("number_or_date")
    if lemmas & NEGATION:
        reasons.add("negation")
    if lemmas & CAUSAL or "escalat" in lower:
        reasons.add("causal_or_consequence")
    if lemmas & COMPARISON or "compared with" in lower:
        reasons.add("comparison")
    if lemmas & TEMPORAL or re.search(r"\b(17\d{2}|18\d{2}|19\d{2}|20\d{2})\b", sent.text):
        reasons.add("temporal")
    if any(token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "attr", "pobj"} for token in sent):
        reasons.add("event_or_relation")
    return sorted(reasons)


def analyze(text: str, model_name: str) -> dict[str, Any]:
    nlp = spacy.load(model_name)
    doc = nlp(text)
    sentences = [sent for sent in doc.sents if sent.text.strip()]
    entities: list[list[Any]] = []
    seen_entities: set[tuple[str, str]] = set()
    numeric: list[list[Any]] = []
    relations: list[list[Any]] = []
    mandatory: list[dict[str, Any]] = []
    for sentence_id, sent in enumerate(sentences):
        reasons = sentence_reason(sent)
        if reasons:
            mandatory.append({"sentence_id": sentence_id, "text": sent.text, "reasons": reasons})
        for ent in sent.ents:
            key = (ent.text, ent.label_)
            if key not in seen_entities:
                seen_entities.add(key)
                entities.append([ent.text, ent.label_])
            if ent.label_ in {"DATE", "TIME", "CARDINAL", "QUANTITY", "PERCENT", "MONEY"}:
                numeric.append([ent.text, ent.label_, sentence_id])
        numeric_matches = list(re.finditer(
            r"\d{3,4}[–-]\d{2,4}", sent.text, flags=re.I
        ))
        numeric_matches.extend(re.finditer(
            r"\b\d[\d,]*(?:\s+(?:million|billion|thousand))?\b",
            sent.text, flags=re.I,
        ))
        for match in numeric_matches:
            if any(
                match.start() >= span.start() and match.end() <= span.end()
                for span in numeric_matches if span is not match
            ):
                continue
            item = [match.group(), "RULE_NUMERIC", sentence_id]
            if not any(
                existing[0] == item[0]
                or item[0] in existing[0]
                for existing in numeric
            ):
                numeric.append(item)
        for subject in [token for token in sent if token.dep_ in {"nsubj", "nsubjpass"}]:
            root = subject.head
            for obj in [child for child in root.children if child.dep_ in {"dobj", "obj", "attr", "pobj", "oprd"}]:
                relations.append([subject.text, root.lemma_, obj.text, sentence_id])
    return {
        "model": model_name,
        "sentences": len(sentences),
        "entities": entities,
        "numeric_temporal": numeric,
        "relations": relations,
        "mandatory_spans": mandatory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["en_core_web_sm", "en_core_web_trf"])
    args = parser.parse_args()
    text = args.input.read_text(encoding="utf-8")
    results: dict[str, Any] = {}
    for model_name in args.models:
        try:
            results[model_name] = analyze(text, model_name)
        except Exception as exc:
            results[model_name] = {"model": model_name, "error": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
