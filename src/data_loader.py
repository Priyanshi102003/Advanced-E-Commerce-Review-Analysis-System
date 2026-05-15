from __future__ import annotations

import bz2
import io
import re
import zipfile
from pathlib import Path
from typing import BinaryIO, Iterable

import pandas as pd

from .config import SENTIMENT_LABELS


FASTTEXT_PATTERN = re.compile(r"^(?P<label>__label__\d+)\s+(?P<text>.*)$")
SUPPORTED_FILE_SUFFIXES = (".txt.bz2", ".bz2", ".txt", ".zip", ".csv")
TEXT_COLUMNS = (
    "review_text",
    "text",
    "review",
    "reviews.text",
    "content",
    "comment",
    "body",
    "summary",
    "title",
    "reviews.title",
)
LABEL_COLUMNS = (
    "sentiment",
    "label",
    "target",
    "polarity",
    "class",
    "rating",
    "reviews.rating",
    "stars",
    "score",
)


def parse_fasttext_line(line: str) -> dict[str, str] | None:
    match = FASTTEXT_PATTERN.match(line.strip())
    if not match:
        return None
    raw_label = match.group("label")
    sentiment = SENTIMENT_LABELS.get(raw_label)
    text = match.group("text").strip()
    if not text or not sentiment:
        return None
    return {"review_text": text, "sentiment": sentiment, "source_label": raw_label}


def _rows_from_fasttext_stream(
    stream: Iterable[str],
    max_rows: int,
    sample_strategy: str = "balanced",
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if max_rows <= 0:
        return rows

    if sample_strategy == "balanced":
        target_per_class = max(1, max_rows // 2)
        counts = {"negative": 0, "positive": 0}
        for line in stream:
            parsed = parse_fasttext_line(line)
            if not parsed:
                continue
            label = parsed["sentiment"]
            if counts.get(label, 0) >= target_per_class:
                if all(value >= target_per_class for value in counts.values()):
                    break
                continue
            rows.append(parsed)
            counts[label] = counts.get(label, 0) + 1
            if len(rows) >= max_rows:
                break
        return rows

    for line in stream:
        parsed = parse_fasttext_line(line)
        if parsed:
            rows.append(parsed)
        if len(rows) >= max_rows:
            break
    return rows


def _read_fasttext_file(path: Path, max_rows: int, sample_strategy: str) -> pd.DataFrame:
    opener = bz2.open if path.suffix.lower() == ".bz2" else open
    with opener(path, mode="rt", encoding="utf-8", errors="ignore") as stream:
        rows = _rows_from_fasttext_stream(stream, max_rows=max_rows, sample_strategy=sample_strategy)
    return pd.DataFrame(rows)


def _normalise_sentiment(value: object, column_name: str | None = None) -> str | None:
    if value is None or pd.isna(value):
        return None
    column_key = (column_name or "").lower()
    rating_like = column_key.split(".")[-1] in {"rating", "stars", "score"}

    if rating_like:
        try:
            rating = float(str(value).strip())
        except ValueError:
            rating = None
        if rating is not None:
            if rating <= 2:
                return "negative"
            if rating >= 4:
                return "positive"
            return None

    if value in SENTIMENT_LABELS:
        return SENTIMENT_LABELS[value]

    text = str(value).strip().lower()
    if text in SENTIMENT_LABELS:
        return SENTIMENT_LABELS[text]
    if "__label__1" in text:
        return "negative"
    if "__label__2" in text:
        return "positive"
    if text in {"negative", "neg", "bad"}:
        return "negative"
    if text in {"positive", "pos", "good"}:
        return "positive"

    try:
        number = float(text)
    except ValueError:
        return None

    if number == 1:
        return "negative"
    if number == 2:
        return "positive"
    if number <= 2:
        return "negative"
    if number >= 4:
        return "positive"
    return None


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _is_supported_data_file(path: Path) -> bool:
    suffixes = "".join(path.suffixes).lower()
    return any(suffixes.endswith(suffix) for suffix in SUPPORTED_FILE_SUFFIXES)


def _find_data_file_in_directory(directory: Path) -> Path:
    candidates = [path for path in directory.rglob("*") if path.is_file() and _is_supported_data_file(path)]
    if not candidates:
        raise FileNotFoundError(f"No supported dataset files found in: {directory}")
    candidates.sort(key=lambda path: ("train" not in path.name.lower(), len(path.parts), path.name.lower()))
    return candidates[0]


def _read_csv(source: str | Path | BinaryIO, max_rows: int) -> pd.DataFrame:
    df = pd.read_csv(source, nrows=max_rows)
    if df.empty:
        return pd.DataFrame(columns=["review_text", "sentiment"])

    text_col = _find_column(df.columns, TEXT_COLUMNS)
    label_col = _find_column(df.columns, LABEL_COLUMNS)

    if text_col is None:
        text_col = df.select_dtypes(include="object").columns[0]

    result = pd.DataFrame({"review_text": df[text_col].astype(str)})
    if label_col:
        result["sentiment"] = df[label_col].map(lambda value: _normalise_sentiment(value, label_col))
    else:
        result["sentiment"] = None

    result = result.dropna(subset=["review_text"])
    result["review_text"] = result["review_text"].str.strip()
    result = result[result["review_text"].str.len() > 0]
    result = result.dropna(subset=["sentiment"])
    return result.reset_index(drop=True)


def _read_zip_bytes(fileobj: BinaryIO, max_rows: int, sample_strategy: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    remaining = max_rows
    with zipfile.ZipFile(fileobj) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".txt", ".txt.bz2", ".bz2", ".csv"))
            and not name.endswith("/")
        ]
        names.sort(key=lambda name: ("train" not in name.lower(), name))
        for name in names:
            if remaining <= 0:
                break
            with archive.open(name) as raw:
                lower_name = name.lower()
                if lower_name.endswith(".csv"):
                    frame = _read_csv(raw, max_rows=remaining)
                else:
                    if lower_name.endswith(".bz2"):
                        text_stream = bz2.open(raw, mode="rt", encoding="utf-8", errors="ignore")
                    else:
                        text_stream = io.TextIOWrapper(raw, encoding="utf-8", errors="ignore")
                    with text_stream:
                        rows = _rows_from_fasttext_stream(
                            text_stream,
                            max_rows=remaining,
                            sample_strategy=sample_strategy,
                        )
                    frame = pd.DataFrame(rows)
            if not frame.empty:
                frames.append(frame)
                remaining -= len(frame)

    if not frames:
        return pd.DataFrame(columns=["review_text", "sentiment"])
    return pd.concat(frames, ignore_index=True).head(max_rows)


def load_reviews_from_path(
    path: str | Path,
    max_rows: int = 6000,
    sample_strategy: str = "balanced",
) -> pd.DataFrame:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    if file_path.is_dir():
        file_path = _find_data_file_in_directory(file_path)

    suffixes = "".join(file_path.suffixes).lower()
    if suffixes.endswith(".zip"):
        with file_path.open("rb") as file:
            return _read_zip_bytes(file, max_rows=max_rows, sample_strategy=sample_strategy)
    if suffixes.endswith(".csv"):
        return _read_csv(file_path, max_rows=max_rows)
    if suffixes.endswith((".txt.bz2", ".bz2", ".txt")):
        return _read_fasttext_file(file_path, max_rows=max_rows, sample_strategy=sample_strategy)

    raise ValueError("Supported dataset formats are .bz2, .txt, .zip, and .csv")


def load_reviews_from_bytes(
    filename: str,
    content: bytes,
    max_rows: int = 6000,
    sample_strategy: str = "balanced",
) -> pd.DataFrame:
    suffixes = "".join(Path(filename).suffixes).lower()
    buffer = io.BytesIO(content)
    if suffixes.endswith(".zip"):
        return _read_zip_bytes(buffer, max_rows=max_rows, sample_strategy=sample_strategy)
    if suffixes.endswith(".csv"):
        return _read_csv(buffer, max_rows=max_rows)
    if suffixes.endswith((".txt.bz2", ".bz2")):
        with bz2.open(buffer, mode="rt", encoding="utf-8", errors="ignore") as stream:
            rows = _rows_from_fasttext_stream(stream, max_rows=max_rows, sample_strategy=sample_strategy)
        return pd.DataFrame(rows)
    if suffixes.endswith(".txt"):
        stream = io.TextIOWrapper(buffer, encoding="utf-8", errors="ignore")
        rows = _rows_from_fasttext_stream(stream, max_rows=max_rows, sample_strategy=sample_strategy)
        return pd.DataFrame(rows)
    raise ValueError("Supported upload formats are .bz2, .txt, .zip, and .csv")


def demo_reviews() -> pd.DataFrame:
    records = [
        ("The headphones arrived early and the sound quality is crisp with solid bass.", "positive"),
        ("Box was crushed and the charger stopped working after two days.", "negative"),
        ("Great value for the price. Setup was simple and everything worked immediately.", "positive"),
        ("Delivery was late, the package was open, and the item looked used.", "negative"),
        ("Excellent build quality and the battery lasts all week.", "positive"),
        ("Cheap material, poor stitching, and it broke during the first use.", "negative"),
        ("Customer support replaced the defective unit quickly. Very happy now.", "positive"),
        ("The seller refused a refund even though the product was damaged.", "negative"),
        ("Perfect fit, authentic product, and careful packaging.", "positive"),
        ("Wrong size was shipped and the return process was confusing.", "negative"),
        ("Instructions were clear and installation took less than five minutes.", "positive"),
        ("Overpriced for such poor quality. I regret buying it.", "negative"),
        ("The glass jars were sealed well and nothing leaked in transit.", "positive"),
        ("Package arrived wet and the contents smelled bad.", "negative"),
        ("This keyboard feels sturdy and the keys are responsive.", "positive"),
        ("Battery overheats while charging and drains very fast.", "negative"),
        ("Original brand item, fast shipping, and worth the money.", "positive"),
        ("Looks counterfeit and the label had an expired date.", "negative"),
        ("The backpack has useful pockets and the zippers feel durable.", "positive"),
        ("Tracking never updated and delivery took three weeks.", "negative"),
        ("Very easy to clean and the product matches the description.", "positive"),
        ("The manual is confusing and setup failed several times.", "negative"),
        ("Good packaging, no dents, and the item works as expected.", "positive"),
        ("The replacement also arrived broken, so quality control seems awful.", "negative"),
        ("Fantastic price for a reliable product.", "positive"),
        ("I received the wrong color and customer service never replied.", "negative"),
        ("Soft fabric, accurate sizing, and comfortable all day.", "positive"),
        ("Too small, rough fabric, and not worth the money.", "negative"),
        ("The blender is powerful, quiet, and simple to use.", "positive"),
        ("It sparked on first use and support has ignored my warranty request.", "negative"),
        ("Five stars, five stars, amazing amazing amazing, buy now!!!", "positive"),
        ("Best thing ever!!! Visit my profile for discount codes!!!", "positive"),
        ("Works well for daily use and feels better than expected.", "positive"),
        ("Dented box, missing parts, and no replacement option.", "negative"),
        ("The screen is bright, performance is smooth, and shipping was quick.", "positive"),
        ("Slow performance, bad battery, and the device freezes often.", "negative"),
        ("Seller shipped the authentic sealed product in two days.", "positive"),
        ("The bottle leaked because the cap was cracked inside the packaging.", "negative"),
        ("Great purchase. The quality, price, and delivery were all excellent.", "positive"),
        ("Do not buy. Terrible quality and impossible to return.", "negative"),
    ]
    return pd.DataFrame(records, columns=["review_text", "sentiment"])


def clean_reviews_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"review_text", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    cleaned = df.copy()
    cleaned["review_text"] = cleaned["review_text"].astype(str).str.strip()
    cleaned["sentiment"] = cleaned["sentiment"].astype(str).str.lower().str.strip()
    cleaned = cleaned[cleaned["review_text"].str.len() > 0]
    cleaned = cleaned[cleaned["sentiment"].isin(["positive", "negative"])]
    return cleaned.drop_duplicates(subset=["review_text", "sentiment"]).reset_index(drop=True)
