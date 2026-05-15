from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .evaluation import confusion_frame, metric_row, report_text
from .preprocessing import NltkTextPreprocessor


MODEL_OPTIONS = {
    "Logistic Regression": lambda random_state: LogisticRegression(
        max_iter=1200,
        class_weight="balanced",
        random_state=random_state,
    ),
    "Naive Bayes": lambda random_state: MultinomialNB(),
    "Random Forest": lambda random_state: RandomForestClassifier(
        n_estimators=120,
        max_depth=80,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    ),
    "Linear SVM": lambda random_state: LinearSVC(
        class_weight="balanced",
        random_state=random_state,
        dual="auto",
    ),
}


@dataclass
class TrainingResult:
    models: dict[str, Pipeline]
    metrics: pd.DataFrame
    reports: dict[str, str]
    confusion_matrices: dict[str, pd.DataFrame]
    predictions: pd.DataFrame
    best_model_name: str | None
    errors: dict[str, str]


def build_pipeline(
    classifier,
    max_features: int,
    ngram_max: int,
    min_df: int,
    lemmatize: bool = False,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("cleaner", NltkTextPreprocessor(lemmatize=lemmatize)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, ngram_max),
                    min_df=min_df,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def _can_stratify(y: pd.Series) -> bool:
    counts = y.value_counts()
    return len(counts) > 1 and bool((counts >= 2).all())


def train_text_classifiers(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    model_names: Iterable[str],
    task_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 12000,
    ngram_max: int = 2,
    min_df: int = 2,
    lemmatize: bool = False,
) -> TrainingResult:
    data = df[[text_col, label_col]].dropna().copy()
    data[text_col] = data[text_col].astype(str)
    data[label_col] = data[label_col].astype(str)
    data = data[data[text_col].str.strip().str.len() > 0]
    data = data[data[label_col].str.strip().str.len() > 0]

    if data[label_col].nunique() < 2:
        return TrainingResult({}, pd.DataFrame(), {}, {}, pd.DataFrame(), None, {"data": "Need at least two classes."})

    stratify = data[label_col] if _can_stratify(data[label_col]) else None
    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    models: dict[str, Pipeline] = {}
    reports: dict[str, str] = {}
    matrices: dict[str, pd.DataFrame] = {}
    prediction_frames: list[pd.DataFrame] = []
    metrics: list[dict[str, float | str]] = []
    errors: dict[str, str] = {}

    for model_name in model_names:
        if model_name not in MODEL_OPTIONS:
            errors[model_name] = "Unknown model option."
            continue
        try:
            pipeline = build_pipeline(
                MODEL_OPTIONS[model_name](random_state),
                max_features=max_features,
                ngram_max=ngram_max,
                min_df=min_df,
                lemmatize=lemmatize,
            )
            pipeline.fit(train_df[text_col], train_df[label_col])
            y_pred = pipeline.predict(test_df[text_col])
            models[model_name] = pipeline
            metrics.append(metric_row(task_name, model_name, test_df[label_col], y_pred))
            reports[model_name] = report_text(test_df[label_col], y_pred)
            matrices[model_name] = confusion_frame(
                test_df[label_col],
                y_pred,
                labels=list(pipeline.named_steps["classifier"].classes_),
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "review_text": test_df[text_col].to_numpy(),
                        "actual": test_df[label_col].to_numpy(),
                        "predicted": y_pred,
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001 - surface model-specific training failures in the UI.
            errors[model_name] = str(exc)

    metrics_df = pd.DataFrame(metrics)
    best_model_name = None
    if not metrics_df.empty:
        best_model_name = (
            metrics_df.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]["model"]
        )

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return TrainingResult(models, metrics_df, reports, matrices, predictions, best_model_name, errors)


def predict_with_confidence(model: Pipeline, texts: list[str]) -> pd.DataFrame:
    """Return predictions with an approximate confidence.

    For probabilistic models, this uses predict_proba.
    For LinearSVC/decision_function models, this converts margins to a [0,1] score.
    """

    predictions = model.predict(texts)
    classifier = model.named_steps["classifier"]
    classes = list(getattr(classifier, "classes_", []))

    confidence: list[float | None] = [None] * len(texts)

    # 1) True probabilities if available.
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(texts)
            confidence = probabilities.max(axis=1).round(3).tolist()
        except Exception:
            confidence = [None] * len(texts)

    # 2) Decision function fallback (e.g., LinearSVC).
    elif hasattr(model, "decision_function"):
        try:
            decision = np.asarray(model.decision_function(texts))
            # Binary case often returns shape (n_samples,)
            if decision.ndim == 1:
                margins = np.abs(decision)
            else:
                # Use the largest margin between best and runner-up as a proxy.
                sorted_margins = np.sort(decision, axis=1)
                margins = np.abs(sorted_margins[:, -1] - sorted_margins[:, -2])

            # Map margin magnitude to [0,1] via a stable logistic.
            # This is not calibrated probability, but provides a monotonic confidence.
            values = 1.0 / (1.0 + np.exp(-margins))
            confidence = values.round(3).tolist()
        except Exception:
            confidence = [None] * len(texts)

    return pd.DataFrame(
        {
            "prediction": predictions,
            "confidence": confidence,
            "classes": [classes] * len(texts),
        }
    )



def top_terms(model: Pipeline, limit: int = 20) -> pd.DataFrame:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    feature_names = np.asarray(vectorizer.get_feature_names_out())

    if hasattr(classifier, "coef_"):
        coefficients = np.asarray(classifier.coef_)
        classes = list(classifier.classes_)
        rows = []
        if coefficients.shape[0] == 1 and len(classes) == 2:
            weights = coefficients[0]
            for index in np.argsort(weights)[-limit:][::-1]:
                rows.append({"class": classes[1], "term": feature_names[index], "weight": float(weights[index])})
            for index in np.argsort(weights)[:limit]:
                rows.append({"class": classes[0], "term": feature_names[index], "weight": float(weights[index])})
        else:
            for class_index, class_name in enumerate(classes):
                weights = coefficients[class_index]
                for index in np.argsort(weights)[-limit:][::-1]:
                    rows.append({"class": class_name, "term": feature_names[index], "weight": float(weights[index])})
        return pd.DataFrame(rows)

    if hasattr(classifier, "feature_log_prob_"):
        log_prob = np.asarray(classifier.feature_log_prob_)
        classes = list(classifier.classes_)
        rows = []
        for class_index, class_name in enumerate(classes):
            for index in np.argsort(log_prob[class_index])[-limit:][::-1]:
                rows.append({"class": class_name, "term": feature_names[index], "weight": float(log_prob[class_index][index])})
        return pd.DataFrame(rows)

    if hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_)
        rows = [
            {"class": "overall", "term": feature_names[index], "weight": float(importances[index])}
            for index in np.argsort(importances)[-limit:][::-1]
        ]
        return pd.DataFrame(rows)

    return pd.DataFrame(columns=["class", "term", "weight"])
