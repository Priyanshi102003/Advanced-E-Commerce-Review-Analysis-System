from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def metric_row(task: str, model_name: str, y_true, y_pred) -> dict[str, float | str]:
    return {
        "task": task,
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def report_text(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, zero_division=0)


def confusion_frame(y_true, y_pred, labels: list[str] | None = None) -> pd.DataFrame:
    if labels is None:
        labels = sorted(pd.Series(list(y_true) + list(y_pred)).dropna().unique())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for actual_index, actual in enumerate(labels):
        for predicted_index, predicted in enumerate(labels):
            rows.append(
                {
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(matrix[actual_index, predicted_index]),
                }
            )
    return pd.DataFrame(rows)
