from __future__ import annotations

import random
from typing import Any, Self

from ._base import _LLMTransformerMixin

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .llm import BaseLLMClient, LLMClient
from .nli import NLIModel
from .sampling import (
    _estimate_n,
    _trim_to_budget,
    sample_texts_kmeans,
    sample_texts_votek,
    sample_texts_within_budget,
)
from .prompts import collection_description as prompts

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


class UnsupervisedTransformer(_LLMTransformerMixin, BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that generates NLI features from unlabeled texts.

    Fits by generating hypotheses (via LLM) that characterize the text collection,
    then scores texts against those hypotheses using an NLI model.

    Fitted attributes:
        features_: Hypotheses as plain strings.

    Example::

        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        pipe = Pipeline([
            ("vect", UnsupervisedTransformer(llm="gpt-4o", nli_model="cross-encoder/nli-deberta-v3-large")),
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
        context_limit: int = 100_000,
        language: str | None = None,
        seed: int | None = None,
        sample_method: str = "random",
        embedding_model: str = "paraphrase-albert-small-v2",
    ) -> None:
        self.llm = llm
        self.nli_model = nli_model
        self.n_features = n_features
        self.context_limit = context_limit
        self.language = language
        self.seed = seed
        self.sample_method = sample_method
        self.embedding_model = embedding_model

    def fit(self, texts: list[str], y=None) -> Self:
        """Generate hypotheses from texts using LLM.

        Args:
            texts: Texts to characterize.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            self
        """
        if self.sample_method not in _SAMPLE_METHODS:
            raise ValueError(f"sample_method must be one of {_SAMPLE_METHODS}, got {self.sample_method!r}")

        _llm = LLMClient(self.llm) if isinstance(self.llm, str) else self.llm
        _rng = random.Random(self.seed)

        budget = self.context_limit - _PROMPT_OVERHEAD
        sampled = _sample_group(
            texts, budget, _llm.count_tokens,
            self.sample_method, self.embedding_model, _rng,
        )

        messages = [
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": prompts.build_user_message(
                sampled, n=self.n_features, language=self.language
            )},
        ]
        result = _llm.complete_json(messages)
        self.features_: list[str] = [
            item["hypothesis"] for item in result.get("features", [])
        ]

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
