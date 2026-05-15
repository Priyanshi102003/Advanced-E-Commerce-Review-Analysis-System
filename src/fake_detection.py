from __future__ import annotations

import math
import re
from dataclasses import dataclass


MARKETING_TERMS = {
    "discount",
    "promo",
    "promotion",
    "sponsored",
    "free product",
    "coupon",
    "buy now",
    "limited offer",
    "visit my profile",
}
GENERIC_EXTREMES = {
    "best ever",
    "life changing",
    "perfect product",
    "five stars",
    "amazing amazing",
    "must buy",
    "do not buy",
    "worst ever",
}


@dataclass(frozen=True)
class FakeReviewSignal:
    label: str
    score: float
    reasons: tuple[str, ...]
    features: dict[str, float]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def score_fake_review(text: str) -> FakeReviewSignal:
    text = "" if text is None else str(text)
    lowered = text.lower()
    tokens = _tokens(text)
    word_count = len(tokens)
    unique_count = len(set(tokens))
    duplicate_ratio = 0.0 if word_count == 0 else 1 - unique_count / word_count
    uppercase_letters = sum(1 for char in text if char.isupper())
    alpha_letters = sum(1 for char in text if char.isalpha())
    uppercase_ratio = 0.0 if alpha_letters == 0 else uppercase_letters / alpha_letters
    exclamation_count = text.count("!")
    punctuation_runs = len(re.findall(r"([!?.,])\1{2,}", text))
    has_url = bool(re.search(r"https?://|www\.|\.com\b", lowered))
    marketing_hits = sum(1 for term in MARKETING_TERMS if term in lowered)
    generic_hits = sum(1 for term in GENERIC_EXTREMES if term in lowered)
    repeated_phrase = bool(re.search(r"\b(\w+)\b(?:\W+\1\b){2,}", lowered))

    score = 0.0
    reasons: list[str] = []

    if word_count < 8:
        score += 0.18
        reasons.append("very short review")
    if duplicate_ratio > 0.32 and word_count >= 8:
        score += min(0.22, duplicate_ratio / 2)
        reasons.append("high word repetition")
    if uppercase_ratio > 0.22 and alpha_letters >= 20:
        score += 0.14
        reasons.append("unusual uppercase emphasis")
    if exclamation_count >= 3:
        score += min(0.16, 0.04 * exclamation_count)
        reasons.append("excessive exclamation marks")
    if punctuation_runs:
        score += 0.08
        reasons.append("repeated punctuation")
    if has_url:
        score += 0.22
        reasons.append("contains URL or external domain")
    if marketing_hits:
        score += min(0.24, 0.12 * marketing_hits)
        reasons.append("promotional language")
    if generic_hits:
        score += min(0.18, 0.09 * generic_hits)
        reasons.append("generic extreme claim")
    if repeated_phrase:
        score += 0.12
        reasons.append("repeated word sequence")

    length_balance = 1 / (1 + math.exp(-(word_count - 20) / 10))
    detail_bonus = 0.08 * length_balance if word_count >= 20 and duplicate_ratio < 0.22 else 0
    score = max(0.0, min(1.0, score - detail_bonus))

    if score >= 0.65:
        label = "high risk"
    elif score >= 0.4:
        label = "review manually"
    else:
        label = "likely genuine"

    if not reasons:
        reasons.append("no strong suspicious text signals")

    return FakeReviewSignal(
        label=label,
        score=round(score, 3),
        reasons=tuple(reasons),
        features={
            "word_count": float(word_count),
            "duplicate_ratio": round(duplicate_ratio, 3),
            "uppercase_ratio": round(uppercase_ratio, 3),
            "exclamation_count": float(exclamation_count),
            "marketing_hits": float(marketing_hits),
            "generic_hits": float(generic_hits),
            "has_url": float(has_url),
        },
    )


def weak_fake_label(text: str) -> str:
    signal = score_fake_review(text)
    return "suspicious" if signal.score >= 0.45 else "likely_genuine"
