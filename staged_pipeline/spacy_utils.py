"""spaCy, rule, compression, and validation helpers for candidate extraction."""

from __future__ import annotations

import re
from typing import Any

ENTITY_LABELS = {
    "PERSON",
    "NORP",
    "FAC",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
}
NUMERIC_LABELS = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}
TRIGGERS = {
    "comparison": {
        "compare",
        "than",
        "more",
        "less",
        "higher",
        "lower",
        "outnumber",
        "versus",
    },
    "cause_consequence": {
        "because",
        "therefore",
        "thus",
        "hence",
        "cause",
        "result",
        "lead",
        "due",
        "escalate",
    },
    "condition": {"if", "unless", "provided", "when", "only"},
    "temporal": {
        "before",
        "after",
        "during",
        "while",
        "later",
        "previously",
        "start",
        "end",
    },
    "contrast": {"but", "however", "although", "whereas", "despite", "instead"},
}
NUMBER_PATTERN = re.compile(
    r"\b\d{3,4}[–-]\d{2,4}\b"
    r"|(?<!\w)[+-]?\d[\d,]*(?:\.\d+)?(?:\s?%|\s+(?:million|billion|thousand))?",
    re.I,
)


def _span_text(token: Any, max_tokens: int = 12) -> str:
    tokens = list(token.subtree)
    if len(tokens) > max_tokens:
        return token.text
    return token.doc[tokens[0].i : tokens[-1].i + 1].text.strip()


def _deduplicate(
    items: list[dict[str, Any]], fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        key = tuple(item.get(field) for field in fields)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def extract_parser_candidates(doc: Any) -> dict[str, Any]:
    """Extract compact, high-recall candidates from a parsed spaCy document."""
    entities: list[dict[str, Any]] = []
    numeric: list[dict[str, Any]] = []
    triggers: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    priority_sentences: list[dict[str, Any]] = []
    sentences = [sentence for sentence in doc.sents if sentence.text.strip()]
    for sentence_id, sentence in enumerate(sentences):
        reasons: set[str] = set()
        for entity in sentence.ents:
            item = {
                "text": entity.text,
                "label": entity.label_,
                "sentence_id": sentence_id,
            }
            if entity.label_ in NUMERIC_LABELS:
                numeric.append(item)
                reasons.add("numeric_temporal")
            elif entity.label_ in ENTITY_LABELS:
                entities.append(item)
                reasons.add("entity")
        for match in NUMBER_PATTERN.finditer(sentence.text):
            if not any(match.group() in item["text"] for item in numeric):
                numeric.append(
                    {
                        "text": match.group(),
                        "label": "RULE_NUMERIC",
                        "sentence_id": sentence_id,
                    }
                )
                reasons.add("numeric_temporal")
        for token in sentence:
            lemma = token.lemma_.lower()
            if token.dep_ == "neg":
                triggers.append(
                    {
                        "text": token.text,
                        "type": "negation",
                        "sentence_id": sentence_id,
                    }
                )
                reasons.add("negation")
            for trigger_type, words in TRIGGERS.items():
                if lemma in words:
                    triggers.append(
                        {
                            "text": token.text,
                            "type": trigger_type,
                            "sentence_id": sentence_id,
                        }
                    )
                    reasons.add(trigger_type)
            if token.pos_ not in {"VERB", "AUX"}:
                continue
            subjects = [
                child
                for child in token.children
                if child.dep_ in {"nsubj", "nsubjpass", "csubj"}
            ]
            objects = [
                child
                for child in token.children
                if child.dep_ in {"dobj", "obj", "iobj", "attr", "oprd", "dative"}
            ]
            for prep in [
                child for child in token.children if child.dep_ in {"prep", "agent"}
            ]:
                objects.extend(child for child in prep.children if child.dep_ == "pobj")
            if subjects and objects:
                reasons.add("relation")
            for subject in subjects:
                for obj in objects:
                    relations.append(
                        {
                            "subject": _span_text(subject),
                            "predicate": token.lemma_,
                            "object": _span_text(obj),
                            "sentence_id": sentence_id,
                            "evidence": sentence.text.strip(),
                        }
                    )
        if reasons:
            priority_sentences.append(
                {
                    "sentence_id": sentence_id,
                    "reasons": sorted(reasons),
                    "evidence": sentence.text.strip(),
                }
            )
    return {
        "entities": _deduplicate(entities, ("text", "label", "sentence_id")),
        "numeric_temporal": _deduplicate(numeric, ("text", "sentence_id")),
        "triggers": _deduplicate(triggers, ("text", "type", "sentence_id")),
        "relations": _deduplicate(
            relations, ("subject", "predicate", "object", "sentence_id")
        ),
        "priority_sentences": priority_sentences,
    }


def compact_parser_candidates(candidates: dict[str, Any]) -> dict[str, Any]:
    """Drop offsets and repeated sentence text before candidates enter the prompt."""
    return {
        "entities": [
            [x["text"], x["label"], x["sentence_id"]] for x in candidates["entities"]
        ],
        "numeric_temporal": [
            [x["text"], x["label"], x["sentence_id"]]
            for x in candidates["numeric_temporal"]
        ],
        "triggers": [
            [x["text"], x["type"], x["sentence_id"]] for x in candidates["triggers"]
        ],
        "relations": [
            [x["subject"], x["predicate"], x["object"], x["sentence_id"]]
            for x in candidates["relations"]
        ],
        "priority_sentences": [
            [x["sentence_id"], x["reasons"]] for x in candidates["priority_sentences"]
        ],
    }


def validate_reorganized(
    parsed: dict[str, Any] | None,
    context: str,
    seeds: dict[str, Any],
    min_coverage: float,
) -> tuple[list[str], dict[str, float]]:
    """Validate schema, evidence substrings, and seed coverage."""
    categories = (
        "events",
        "entities",
        "quantities",
        "triggers",
        "relations",
        "priority_sentences",
    )
    if not isinstance(parsed, dict) or set(parsed) != set(categories):
        return [f"output must contain exactly {list(categories)}"], {}
    errors: list[str] = []
    output_items: dict[str, list[dict[str, Any]]] = {}
    for category in categories:
        items = parsed.get(category)
        if not isinstance(items, list):
            errors.append(f"{category} must be a list")
            continue
        output_items[category] = [item for item in items if isinstance(item, dict)]
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"{category}[{index}] must be an object")
                continue
            evidence = item.get("evidence")
            if (
                not isinstance(evidence, str)
                or not evidence.strip()
                or evidence not in context
            ):
                errors.append(
                    f"{category}[{index}].evidence is not a context substring"
                )
            if category == "relations":
                for field in ("subject", "predicate", "object"):
                    if not isinstance(item.get(field), str) or not item[field].strip():
                        errors.append(f"{category}[{index}].{field} must be non-empty")
            elif category == "priority_sentences":
                if (
                    not isinstance(item.get("sentence_id"), int)
                    or not isinstance(item.get("reasons"), list)
                    or not all(isinstance(reason, str) for reason in item["reasons"])
                ):
                    errors.append(
                        f"{category}[{index}] requires sentence_id and reasons"
                    )
            else:
                if not isinstance(item.get("text"), str) or not item["text"].strip():
                    errors.append(f"{category}[{index}].text must be non-empty")
                if category in {"entities", "quantities", "triggers"} and (
                    not isinstance(item.get("type"), str) or not item["type"].strip()
                ):
                    errors.append(f"{category}[{index}].type must be non-empty")

    coverage: dict[str, float] = {}
    category_mapping = {
        "entities": ("entities", "events"),
        "numeric_temporal": ("quantities", "events"),
        "triggers": ("triggers",),
    }
    for category, output_categories in category_mapping.items():
        source = seeds.get(category, [])
        candidates = [
            item
            for output_category in output_categories
            for item in output_items.get(output_category, [])
        ]
        covered = sum(
            1
            for item in source
            if any(
                item["text"].casefold() in str(candidate.get("text", "")).casefold()
                for candidate in candidates
            )
        )
        coverage[category] = covered / len(source) if source else 1.0
    source_relations = seeds.get("relations", [])
    output_relations = [
        item for item in parsed.get("relations", []) if isinstance(item, dict)
    ]
    coverage["relations"] = (
        sum(
            1
            for source in source_relations
            if any(
                source.get("evidence", "") in candidate.get("evidence", "")
                or candidate.get("evidence", "") in source.get("evidence", "")
                or (
                    source.get("subject", "").casefold()
                    in candidate.get("subject", "").casefold()
                    and source.get("object", "").casefold()
                    in candidate.get("object", "").casefold()
                )
                for candidate in output_relations
            )
        )
        / len(source_relations)
        if source_relations
        else 1.0
    )
    output_priority = {
        item.get("sentence_id")
        for item in parsed.get("priority_sentences", [])
        if isinstance(item, dict)
    }
    source_priority = seeds.get("priority_sentences", [])
    coverage["priority_sentences"] = (
        sum(1 for item in source_priority if item["sentence_id"] in output_priority)
        / len(source_priority)
        if source_priority
        else 1.0
    )
    for category, score in coverage.items():
        if score < min_coverage:
            errors.append(
                f"{category} coverage {score:.3f} is below {min_coverage:.3f}"
            )
    return errors, coverage
