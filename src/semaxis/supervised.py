from __future__ import annotations

import random
from itertools import combinations
from typing import Any, NamedTuple, Self

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from .llm import BaseLLMClient, LLMClient
from .nli import NLIModel
from .sampling import (
    _estimate_n,
    _trim_to_budget,
    sample_texts_kmeans,
    sample_texts_votek,
    sample_texts_within_budget,
)
from .prompts import discriminative_features as prompts

_SAMPLE_METHODS = ("random", "kmeans", "votek")


def _sample_group(
    texts: list[str],
    budget: int,
    tokenizer_fn: Any,
    method: str,
    embedding_model: str,
    rng: random.Random,
) -> list[str]:
    if method == "random":
        return sample_texts_within_budget(texts, budget, tokenizer_fn, rng)

    from sentence_transformers import SentenceTransformer
    embeddings = SentenceTransformer(embedding_model).encode(
        texts, show_progress_bar=False, convert_to_numpy=True
    )
    n = _estimate_n(texts, budget, tokenizer_fn)
    if method == "kmeans":
        sampled = sample_texts_kmeans(texts, n, embeddings, rng)
    else:
        sampled = sample_texts_votek(texts, n, embeddings, rng=rng)
    return _trim_to_budget(sampled, budget, tokenizer_fn)

_PROMPT_OVERHEAD = 500


class FeatureMeta(NamedTuple):
    positive: Any
    negative: Any


class SupervisedTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that generates discriminative NLI features.

    Fits by generating hypotheses (via LLM) that distinguish between classes,
    then scores texts against those hypotheses using an NLI model.

    Supports binary and multi-class labels (numeric or string).
    For multi-class, use ``strategy="ovr"`` (one-vs-rest) or ``strategy="ovo"``
    (one-vs-one). Binary classification ignores ``strategy``.

    Fitted attributes:
        classes_: Unique class labels in sorted order.
        features_: Hypotheses as plain strings, parallel to ``feature_meta_``.
        feature_meta_: FeatureMeta(positive, negative) for each hypothesis,
            where negative is the original class label or ``"rest"`` for OvR.

    Example::

        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        pipe = Pipeline([
            ("vect", SupervisedTransformer(llm="gpt-4o", nli_model="cross-encoder/nli-deberta-v3-large")),
            ("clf", LogisticRegression()),
        ])
        cross_val_score(pipe, texts, labels, cv=5)

    Note:
        When ``llm`` is a non-picklable client instance (e.g. ``LLMClient``),
        parallel execution via joblib's loky backend (``n_jobs != 1``) will
        still fail because joblib pickles estimators for subprocess dispatch.
        Use ``n_jobs=1`` or the threading backend::

            import joblib
            with joblib.parallel_backend("threading"):
                cross_val_score(pipe, texts, labels, cv=5, n_jobs=-1)
    """

    def __init__(
        self,
        llm: BaseLLMClient | str,
        nli_model: str = "cross-encoder/nli-deberta-v3-large",
        n_features: int = 20,
        strategy: str = "ovr",
        context_limit: int = 100_000,
        language: str | None = None,
        seed: int | None = None,
        sample_method: str = "random",
        embedding_model: str = "paraphrase-albert-small-v2",
    ) -> None:
        self.llm = llm
        self.nli_model = nli_model
        self.n_features = n_features
        self.strategy = strategy
        self.context_limit = context_limit
        self.language = language
        self.seed = seed
        self.sample_method = sample_method
        self.embedding_model = embedding_model

    def __sklearn_clone__(self) -> Self:
        return type(self)(**self.get_params(deep=False))

    def fit(self, texts: list[str], y: Any) -> Self:
        """Generate discriminative hypotheses from training texts and labels.

        Args:
            texts: Training texts.
            y: Class labels (numeric or string).

        Returns:
            self
        """
        if self.strategy not in ("ovr", "ovo"):
            raise ValueError(f"strategy must be 'ovr' or 'ovo', got {self.strategy!r}")
        if self.sample_method not in _SAMPLE_METHODS:
            raise ValueError(f"sample_method must be one of {_SAMPLE_METHODS}, got {self.sample_method!r}")

        _llm = LLMClient(self.llm) if isinstance(self.llm, str) else self.llm
        _rng = random.Random(self.seed)

        le = LabelEncoder()
        y_enc: np.ndarray = le.fit_transform(y)
        self.classes_ = le.classes_

        n_classes = len(self.classes_)
        budget = self.context_limit - _PROMPT_OVERHEAD

        if n_classes == 2:
            pairs: list[tuple[int, int | str]] = [(0, 1)]
        elif self.strategy == "ovr":
            pairs = [(i, "rest") for i in range(n_classes)]
        else:
            pairs = list(combinations(range(n_classes), 2))

        self.features_: list[str] = []
        self.feature_meta_: list[FeatureMeta] = []

        for pos_idx, neg_idx in pairs:
            pos_label = self.classes_[pos_idx]
            pos_texts = [t for t, yi in zip(texts, y_enc) if yi == pos_idx]

            if neg_idx == "rest":
                neg_texts = [t for t, yi in zip(texts, y_enc) if yi != pos_idx]
                neg_label: Any = "rest"
            else:
                neg_texts = [t for t, yi in zip(texts, y_enc) if yi == neg_idx]
                neg_label = self.classes_[neg_idx]

            pos_sampled = _sample_group(
                pos_texts, budget // 2, _llm.count_tokens,
                self.sample_method, self.embedding_model, _rng,
            )
            neg_sampled = _sample_group(
                neg_texts, budget // 2, _llm.count_tokens,
                self.sample_method, self.embedding_model, _rng,
            )

            messages = [
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": prompts.build_user_message(
                    pos_texts=pos_sampled,
                    neg_texts=neg_sampled,
                    pos_label=str(pos_label),
                    neg_label=str(neg_label),
                    n=self.n_features,
                    language=self.language,
                )},
            ]
            result = _llm.complete_json(messages)
            hypotheses = [item["hypothesis"] for item in result.get("features", [])]

            self.features_.extend(hypotheses)
            self.feature_meta_.extend(
                FeatureMeta(positive=pos_label, negative=neg_label)
                for _ in hypotheses
            )

        self._nli = NLIModel(self.nli_model)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Score texts against fitted hypotheses using NLI.

        Args:
            texts: Texts to score.

        Returns:
            np.ndarray of shape (n_texts, n_features) with entailment scores in [0, 1].
        """
        columns = [
            self._nli.score(texts, [h] * len(texts))
            for h in self.features_
        ]
        return np.column_stack(columns)
