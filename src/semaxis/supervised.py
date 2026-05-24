from __future__ import annotations

import random
from itertools import combinations
from typing import Any, NamedTuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from .llm import BaseLLMClient, LLMClient
from .nli import NLIModel
from .sampling import sample_texts_within_budget
from .prompts import discriminative_features as prompts

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
    ) -> None:
        self.llm = llm
        self.nli_model = nli_model
        self.n_features = n_features
        self.strategy = strategy
        self.context_limit = context_limit
        self.language = language
        self.seed = seed

    def fit(self, texts: list[str], y: Any) -> SupervisedTransformer:
        """Generate discriminative hypotheses from training texts and labels.

        Args:
            texts: Training texts.
            y: Class labels (numeric or string).

        Returns:
            self
        """
        if self.strategy not in ("ovr", "ovo"):
            raise ValueError(f"strategy must be 'ovr' or 'ovo', got {self.strategy!r}")

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

            pos_sampled = sample_texts_within_budget(
                pos_texts, budget // 2, _llm.count_tokens, rng=_rng
            )
            neg_sampled = sample_texts_within_budget(
                neg_texts, budget // 2, _llm.count_tokens, rng=_rng
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
