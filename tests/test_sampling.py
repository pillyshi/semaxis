"""Unit tests for embedding-aware sampling functions."""
import random

import numpy as np

from semaxis.sampling import (
    _estimate_n,
    _trim_to_budget,
    sample_texts_kmeans,
    sample_texts_votek,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _embeddings(n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, dim)).astype(np.float32)


TEXTS = [f"text {i}" for i in range(20)]
EMBEDDINGS = _embeddings(20)


# ---------------------------------------------------------------------------
# sample_texts_kmeans
# ---------------------------------------------------------------------------

class TestSampleTextsKmeans:
    def test_returns_correct_count(self):
        result = sample_texts_kmeans(TEXTS, 5, EMBEDDINGS)
        assert len(result) == 5

    def test_returns_subset_of_input(self):
        result = sample_texts_kmeans(TEXTS, 5, EMBEDDINGS)
        assert all(t in TEXTS for t in result)

    def test_no_duplicates(self):
        result = sample_texts_kmeans(TEXTS, 5, EMBEDDINGS)
        assert len(result) == len(set(result))

    def test_n_larger_than_texts_returns_all(self):
        result = sample_texts_kmeans(TEXTS, 100, EMBEDDINGS)
        assert set(result) == set(TEXTS)

    def test_n_zero_returns_empty(self):
        assert sample_texts_kmeans(TEXTS, 0, EMBEDDINGS) == []

    def test_n_equals_len_returns_all(self):
        result = sample_texts_kmeans(TEXTS, len(TEXTS), EMBEDDINGS)
        assert set(result) == set(TEXTS)

    def test_reproducible_with_rng(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        r1 = sample_texts_kmeans(TEXTS, 5, EMBEDDINGS, rng=rng1)
        r2 = sample_texts_kmeans(TEXTS, 5, EMBEDDINGS, rng=rng2)
        assert r1 == r2

    def test_single_text(self):
        result = sample_texts_kmeans(["only"], 1, _embeddings(1))
        assert result == ["only"]


# ---------------------------------------------------------------------------
# sample_texts_votek
# ---------------------------------------------------------------------------

class TestSampleTextsVotek:
    def test_returns_correct_count(self):
        result = sample_texts_votek(TEXTS, 5, EMBEDDINGS)
        assert len(result) == 5

    def test_returns_subset_of_input(self):
        result = sample_texts_votek(TEXTS, 5, EMBEDDINGS)
        assert all(t in TEXTS for t in result)

    def test_no_duplicates(self):
        result = sample_texts_votek(TEXTS, 5, EMBEDDINGS)
        assert len(result) == len(set(result))

    def test_n_larger_than_texts_returns_all(self):
        result = sample_texts_votek(TEXTS, 100, EMBEDDINGS)
        assert set(result) == set(TEXTS)

    def test_n_zero_returns_empty(self):
        assert sample_texts_votek(TEXTS, 0, EMBEDDINGS) == []

    def test_n_equals_len_returns_all(self):
        result = sample_texts_votek(TEXTS, len(TEXTS), EMBEDDINGS)
        assert set(result) == set(TEXTS)

    def test_k_larger_than_n_texts_minus_1(self):
        small_texts = TEXTS[:3]
        small_emb = EMBEDDINGS[:3]
        result = sample_texts_votek(small_texts, 2, small_emb, k=100)
        assert len(result) == 2
        assert all(t in small_texts for t in result)

    def test_single_text(self):
        result = sample_texts_votek(["only"], 1, _embeddings(1))
        assert result == ["only"]


# ---------------------------------------------------------------------------
# _estimate_n
# ---------------------------------------------------------------------------

class TestEstimateN:
    def test_basic(self):
        tokenizer_fn = lambda t: 5  # noqa: E731
        n = _estimate_n(["a", "b", "c"], 50, tokenizer_fn)
        assert n == 10  # 50 / 5

    def test_at_least_one(self):
        tokenizer_fn = lambda t: 10_000  # noqa: E731
        n = _estimate_n(["very long text"], 1, tokenizer_fn)
        assert n == 1

    def test_empty_texts(self):
        tokenizer_fn = lambda t: 5  # noqa: E731
        n = _estimate_n([], 100, tokenizer_fn)
        assert n == 0


# ---------------------------------------------------------------------------
# _trim_to_budget
# ---------------------------------------------------------------------------

class TestTrimToBudget:
    def test_all_fit(self):
        tokenizer_fn = lambda t: 1  # noqa: E731
        result = _trim_to_budget(["a", "b", "c"], 10, tokenizer_fn)
        assert result == ["a", "b", "c"]

    def test_some_trimmed(self):
        tokenizer_fn = lambda t: 3  # noqa: E731
        result = _trim_to_budget(["a", "b", "c", "d"], 8, tokenizer_fn)
        assert result == ["a", "b"]

    def test_empty_input(self):
        tokenizer_fn = lambda t: 1  # noqa: E731
        assert _trim_to_budget([], 10, tokenizer_fn) == []

    def test_first_text_exceeds_budget(self):
        tokenizer_fn = lambda t: 100  # noqa: E731
        assert _trim_to_budget(["big"], 10, tokenizer_fn) == []
