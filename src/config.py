from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_TRAIN_FILE = RAW_DATA_DIR / "train.ft.txt.bz2"
DEFAULT_TEST_FILE = RAW_DATA_DIR / "test.ft.txt.bz2"

SENTIMENT_LABELS = {
    "__label__1": "negative",
    "__label__2": "positive",
    "1": "negative",
    "2": "positive",
    1: "negative",
    2: "positive",
}


@dataclass(frozen=True)
class AppDefaults:
    max_rows: int = 6000
    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 12000
    min_df: int = 2
    ngram_max: int = 2


DEFAULTS = AppDefaults()
