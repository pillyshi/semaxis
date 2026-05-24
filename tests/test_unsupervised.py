"""Unit tests for UnsupervisedTransformer."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from prism import UnsupervisedTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm(n_features: int = 3) -> MagicMock:
    llm = MagicMock()
    llm.count_tokens.return_value = 1
    llm.complete_json.return_value = {
        "features": [{"hypothesis": f"hyp {i}"} for i in range(n_features)]
    }
    return llm


def _make_nli(score_value: float = 0.8) -> MagicMock:
    nli = MagicMock()
    nli.score.side_effect = lambda texts, hypotheses: np.full(len(texts), score_value)
    return nli


def _fit(
    transformer: UnsupervisedTransformer,
    llm: MagicMock,
    nli: MagicMock,
    texts: list[str] | None = None,
) -> UnsupervisedTransformer:
    if texts is None:
        texts = ["text 1", "text 2", "text 3"]
    with patch("prism.unsupervised.NLIModel", return_value=nli):
        transformer.llm = llm
        return transformer.fit(texts)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def test_fit_returns_self():
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=3)
    result = _fit(t, _make_llm(3), _make_nli())
    assert result is t


def test_fit_sets_features():
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=3)
    _fit(t, _make_llm(3), _make_nli())
    assert t.features_ == ["hyp 0", "hyp 1", "hyp 2"]


def test_fit_features_are_strings():
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=3)
    _fit(t, _make_llm(3), _make_nli())
    assert all(isinstance(f, str) for f in t.features_)


def test_fit_y_is_optional():
    """fit(texts) must work without passing y."""
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    llm = _make_llm(2)
    nli = _make_nli()
    with patch("prism.unsupervised.NLIModel", return_value=nli):
        t.llm = llm
        t.fit(["a", "b", "c"])  # no y


def test_fit_y_ignored():
    """Passing y should not affect features_."""
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    llm = _make_llm(2)
    nli = _make_nli()
    with patch("prism.unsupervised.NLIModel", return_value=nli):
        t.llm = llm
        t.fit(["a", "b", "c"], y=[0, 1, 0])
    assert t.features_ == ["hyp 0", "hyp 1"]


def test_fit_calls_llm_once():
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=3)
    llm = _make_llm(3)
    _fit(t, llm, _make_nli())
    assert llm.complete_json.call_count == 1


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------

def test_transform_shape():
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=4)
    _fit(t, _make_llm(4), _make_nli())
    X = t.transform(["x", "y", "z"])
    assert X.shape == (3, 4)


def test_transform_values():
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=3)
    _fit(t, _make_llm(3), _make_nli(0.6))
    X = t.transform(["x", "y"])
    np.testing.assert_allclose(X, 0.6)


def test_transform_uses_all_features():
    """NLI is called once per hypothesis in features_."""
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=5)
    nli = _make_nli()
    _fit(t, _make_llm(5), nli)
    t.transform(["x", "y"])
    assert nli.score.call_count == 5


def test_transform_different_texts_from_fit():
    """transform can be called on texts not seen during fit."""
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    _fit(t, _make_llm(2), _make_nli(0.5))
    X = t.transform(["unseen text A", "unseen text B", "unseen text C"])
    assert X.shape == (3, 2)
