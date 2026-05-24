from __future__ import annotations

import random

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .llm import BaseLLMClient, LLMClient
from .nli import NLIModel
from .sampling import sample_texts_within_budget
from .prompts import collection_description as prompts

_PROMPT_OVERHEAD = 500


class UnsupervisedTransformer(BaseEstimator, TransformerMixin):
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
    """

    def __init__(
        self,
        llm: BaseLLMClient | str,
        nli_model: str = "cross-encoder/nli-deberta-v3-large",
        n_features: int = 20,
        context_limit: int = 100_000,
        language: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.llm = llm
        self.nli_model = nli_model
        self.n_features = n_features
        self.context_limit = context_limit
        self.language = language
        self.seed = seed

    def fit(self, texts: list[str], y=None) -> UnsupervisedTransformer:
        """Generate hypotheses from texts using LLM.

        Args:
            texts: Texts to characterize.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            self
        """
        _llm = LLMClient(self.llm) if isinstance(self.llm, str) else self.llm
        _rng = random.Random(self.seed)

        budget = self.context_limit - _PROMPT_OVERHEAD
        sampled = sample_texts_within_budget(texts, budget, _llm.count_tokens, rng=_rng)

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
