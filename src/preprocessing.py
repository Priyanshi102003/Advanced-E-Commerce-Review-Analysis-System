from __future__ import annotations

import re
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.base import BaseEstimator, TransformerMixin


def ensure_nltk_resources(use_lemmatizer: bool = False) -> None:
    resources = [("corpora/stopwords", "stopwords")]
    if use_lemmatizer:
        resources.extend(
            [
                ("corpora/wordnet", "wordnet"),
                ("corpora/omw-1.4", "omw-1.4"),
            ]
        )
    for resource_path, package in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)


class NltkTextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = False,
        min_token_length: int = 2,
    ) -> None:
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_length = min_token_length

    def fit(self, X: Iterable[str], y: Iterable[str] | None = None) -> "NltkTextPreprocessor":
        ensure_nltk_resources(use_lemmatizer=self.lemmatize)
        self._tokenizer = RegexpTokenizer(r"[A-Za-z][A-Za-z']+")
        self._stop_words = set(stopwords.words("english")) if self.remove_stopwords else set()
        self._lemmatizer = WordNetLemmatizer() if self.lemmatize else None
        return self

    def transform(self, X: Iterable[str]) -> list[str]:
        if not hasattr(self, "_tokenizer"):
            self.fit([])
        cleaned_texts: list[str] = []
        for value in X:
            text = "" if value is None else str(value)
            if self.lowercase:
                text = text.lower()
            text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
            text = re.sub(r"\d+", " numbertoken ", text)
            tokens = self._tokenizer.tokenize(text)
            processed_tokens: list[str] = []
            for token in tokens:
                if len(token) < self.min_token_length:
                    continue
                if token in self._stop_words:
                    continue
                if self._lemmatizer:
                    token = self._lemmatizer.lemmatize(token)
                processed_tokens.append(token)
            cleaned_texts.append(" ".join(processed_tokens))
        return cleaned_texts
