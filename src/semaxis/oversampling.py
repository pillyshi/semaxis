from __future__ import annotations

import random
from typing import Any, Self

from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator

from ._base import _LLMTransformerMixin
from .llm import BaseLLMClient, LLMClient
from .supervised import _PROMPT_OVERHEAD, _SAMPLE_METHODS, _sample_group
from .prompts import hard_positive as prompts


class HardPositive(BaseModel):
    text: str
    positive_evidence: list[str]
    confusing_evidence: list[str]


class BoundaryFeature(BaseModel):
    feature: str
    importance: float = Field(ge=0.0, le=1.0)


class HardPositiveGenerationResult(BaseModel):
    positive_features: list[str]
    negative_features: list[str]
    boundary_features: list[BoundaryFeature]
    hard_positives: list[HardPositive]


class HardPositiveOverSampler(_LLMTransformerMixin, BaseEstimator):
    """LLM-based oversampler that generates hard positives for binary classification.

    Implements the imbalanced-learn ``fit_resample(X, y)`` interface for text arrays.
    Unlike standard oversamplers, ``X`` is a list of raw strings, not a numeric feature
    matrix. Only binary labels (0/1) are supported.

    The LLM analyzes positive and negative examples to identify boundary-defining
    features, then synthesizes texts that carry all essential positive-class criteria
    while being superficially ambiguous — texts that experts would label positive but
    that shallow classifiers or untrained humans might label negative.

    Fitted attributes:
        generation_result_: Full LLM response including feature analysis and
            per-sample evidence. Useful for auditing boundary-defining features
            and why each generated text is considered a hard positive.

    Example::

        from imblearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression

        sampler = HardPositiveOverSampler(llm="gpt-4o", n_synthesized=20)
        X_aug, y_aug = sampler.fit_resample(texts, labels)

        for hp in sampler.generation_result_.hard_positives:
            print(hp.text)
            print("  evidence:", hp.positive_evidence)
    """

    def __init__(
        self,
        llm: BaseLLMClient | str,
        n_synthesized: int = 10,
        context_limit: int = 100_000,
        seed: int | None = None,
        sample_method: str = "random",
        embedding_model: str = "paraphrase-albert-small-v2",
    ) -> None:
        self.llm = llm
        self.n_synthesized = n_synthesized
        self.context_limit = context_limit
        self.seed = seed
        self.sample_method = sample_method
        self.embedding_model = embedding_model

    def fit_resample(
        self,
        X: list[str],
        y: Any,
    ) -> tuple[list[str], list[int]]:
        """Generate hard positives and append them to the dataset.

        Args:
            X: Raw texts. Must be a list of strings, not a numeric feature matrix.
            y: Binary labels. Positive class must be 1; negative class must be 0.

        Returns:
            Tuple of (augmented texts, augmented labels) where the appended texts
            are the generated hard positives and the appended labels are all 1.
        """
        if self.sample_method not in _SAMPLE_METHODS:
            raise ValueError(
                f"sample_method must be one of {_SAMPLE_METHODS}, got {self.sample_method!r}"
            )

        y_list = list(y)
        if len(X) != len(y_list):
            raise ValueError(
                f"X and y must have the same length, got len(X)={len(X)} and len(y)={len(y_list)}"
            )
        label_set = set(y_list)
        if not label_set <= {0, 1}:
            raise ValueError(
                f"y must contain only binary labels {{0, 1}}, got {label_set - {0, 1}}"
            )

        _llm = LLMClient(self.llm) if isinstance(self.llm, str) else self.llm
        _rng = random.Random(self.seed)

        pos_texts = [t for t, yi in zip(X, y_list) if yi == 1]
        neg_texts = [t for t, yi in zip(X, y_list) if yi == 0]

        budget = (self.context_limit - _PROMPT_OVERHEAD) // 2
        pos_sampled = _sample_group(
            pos_texts, budget, _llm.count_tokens,
            self.sample_method, self.embedding_model, _rng,
        )
        neg_sampled = _sample_group(
            neg_texts, budget, _llm.count_tokens,
            self.sample_method, self.embedding_model, _rng,
        )

        messages = [
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": prompts.build_user_message(
                pos_texts=pos_sampled,
                neg_texts=neg_sampled,
                n_synthesized=self.n_synthesized,
            )},
        ]
        result = _llm.complete_json(messages)
        self.generation_result_ = HardPositiveGenerationResult.model_validate(result)

        generated_texts = [hp.text for hp in self.generation_result_.hard_positives]
        X_aug = list(X) + generated_texts
        y_aug = y_list + [1] * len(generated_texts)
        return X_aug, y_aug
