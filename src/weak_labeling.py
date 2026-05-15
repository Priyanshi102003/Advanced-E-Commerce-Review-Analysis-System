from __future__ import annotations

import re
from dataclasses import dataclass


ISSUE_TAXONOMY: dict[str, tuple[str, ...]] = {
    "delivery": (
        "delivery",
        "delivered",
        "shipping",
        "shipment",
        "late",
        "delayed",
        "courier",
        "tracking",
        "arrived",
        "lost",
    ),
    "quality": (
        "quality",
        "broken",
        "defective",
        "damaged",
        "stopped working",
        "poorly made",
        "cheap material",
        "durable",
        "sturdy",
        "cracked",
    ),
    "packaging": (
        "packaging",
        "package",
        "box",
        "packed",
        "sealed",
        "bubble wrap",
        "dented",
        "crushed",
        "leaked",
    ),
    "fit_size": (
        "size",
        "fit",
        "too small",
        "too large",
        "dimensions",
        "compatible",
        "wrong size",
        "wrong color",
    ),
    "price_value": (
        "price",
        "value",
        "worth",
        "money",
        "overpriced",
        "deal",
        "expensive",
        "cheap",
    ),
    "support_returns": (
        "customer service",
        "support",
        "seller",
        "refund",
        "return",
        "replacement",
        "warranty",
        "replied",
    ),
    "authenticity": (
        "fake",
        "counterfeit",
        "authentic",
        "original",
        "knockoff",
        "expired",
        "label",
    ),
    "usability": (
        "easy to use",
        "setup",
        "install",
        "installation",
        "instructions",
        "manual",
        "confusing",
        "clean",
    ),
    "performance": (
        "battery",
        "charge",
        "charging",
        "performance",
        "speed",
        "slow",
        "overheats",
        "freezes",
        "responsive",
    ),
}


@dataclass(frozen=True)
class IssuePrediction:
    label: str
    score: int
    matches: tuple[str, ...]


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    escaped = re.escape(keyword.lower())
    if " " in keyword:
        return re.compile(escaped, flags=re.IGNORECASE)
    return re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)


COMPILED_TAXONOMY = {
    label: tuple((keyword, _keyword_pattern(keyword)) for keyword in keywords)
    for label, keywords in ISSUE_TAXONOMY.items()
}


def classify_issue(text: str) -> IssuePrediction:
    text = "" if text is None else str(text).lower()
    scores: dict[str, int] = {}
    matches: dict[str, list[str]] = {}
    for label, keyword_patterns in COMPILED_TAXONOMY.items():
        for keyword, pattern in keyword_patterns:
            hits = pattern.findall(text)
            if hits:
                scores[label] = scores.get(label, 0) + len(hits)
                matches.setdefault(label, []).append(keyword)

    if not scores:
        return IssuePrediction(label="general", score=0, matches=())

    best_label = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return IssuePrediction(
        label=best_label,
        score=scores[best_label],
        matches=tuple(sorted(set(matches.get(best_label, [])))),
    )


def issue_display_name(label: str) -> str:
    return label.replace("_", " ").title()
