"""Unit tests for UnsupervisedTransformer."""
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from semaxis import UnsupervisedTransformer


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
    with patch("semaxis.unsupervised.NLIModel", return_value=nli):
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
    with patch("semaxis.unsupervised.NLIModel", return_value=nli):
        t.llm = llm
        t.fit(["a", "b", "c"])  # no y


def test_fit_y_ignored():
    """Passing y should not affect features_."""
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    llm = _make_llm(2)
    nli = _make_nli()
    with patch("semaxis.unsupervised.NLIModel", return_value=nli):
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


# ---------------------------------------------------------------------------
# sample_method parameter
# ---------------------------------------------------------------------------

def test_sample_method_invalid_raises():
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m", sample_method="bad")
    with pytest.raises(ValueError, match="sample_method"):
        _fit(t, _make_llm(2), _make_nli())


def test_sample_method_random_default():
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m")
    assert t.sample_method == "random"


def _fit_with_method(method: str) -> UnsupervisedTransformer:
    t = UnsupervisedTransformer(
        llm=MagicMock(), nli_model="m", n_features=2, sample_method=method
    )
    llm = _make_llm(2)
    nli = _make_nli()
    fake_embeddings = np.zeros((3, 4))
    fake_model = MagicMock()
    fake_model.encode.return_value = fake_embeddings
    with patch("semaxis.unsupervised.NLIModel", return_value=nli), \
         patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        t.llm = llm
        t.fit(["text 1", "text 2", "text 3"])
    return t


def test_sample_method_kmeans_fits_without_error():
    t = _fit_with_method("kmeans")
    assert t.features_ == ["hyp 0", "hyp 1"]


def test_sample_method_votek_fits_without_error():
    t = _fit_with_method("votek")
    assert t.features_ == ["hyp 0", "hyp 1"]


# ---------------------------------------------------------------------------
# sklearn clone compatibility
# ---------------------------------------------------------------------------

def test_sklearn_clone_with_llm_client_instance():
    """clone() must not raise even when llm holds a non-picklable object."""
    llm = MagicMock()
    llm._lock = threading.RLock()  # simulate unpicklable state
    t = UnsupervisedTransformer(llm=llm, nli_model="m", n_features=3)
    cloned = clone(t)
    assert cloned is not t
    assert cloned.llm is llm


def test_sklearn_clone_preserves_params():
    llm = MagicMock()
    t = UnsupervisedTransformer(
        llm=llm, nli_model="my-nli", n_features=7,
        context_limit=50_000, language="en", seed=42,
        sample_method="kmeans", embedding_model="my-embed",
    )
    cloned = clone(t)
    assert cloned.get_params(deep=False) == t.get_params(deep=False)


def test_sklearn_clone_with_string_llm():
    t = UnsupervisedTransformer(llm="gpt-4o", nli_model="m")
    cloned = clone(t)
    assert cloned.llm == "gpt-4o"


def test_sklearn_clone_preserves_sklearn_output_config():
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m")
    t._sklearn_output_config = {"transform": "pandas"}
    cloned = clone(t)
    assert cloned._sklearn_output_config == {"transform": "pandas"}


def test_sklearn_clone_preserves_metadata_request():
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m")
    t._metadata_request = {"key": "value"}
    cloned = clone(t)
    assert cloned._metadata_request == {"key": "value"}
    assert cloned._metadata_request is not t._metadata_request  # deep-copied


def test_sample_method_kmeans_calls_sentence_transformer():
    llm = _make_llm(2)
    nli = _make_nli()
    fake_embeddings = np.zeros((3, 4))
    fake_model = MagicMock()
    fake_model.encode.return_value = fake_embeddings

    t = UnsupervisedTransformer(
        llm=MagicMock(), nli_model="m", n_features=2, sample_method="kmeans"
    )
    with patch("semaxis.unsupervised.NLIModel", return_value=nli), \
         patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as MockST:
        t.llm = llm
        t.fit(["text 1", "text 2", "text 3"])
        assert MockST.called


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def test_save_before_fit_raises(tmp_path):
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m")
    with pytest.raises(NotFittedError):
        t.save(tmp_path / "model.json")


def test_save_load_roundtrip(tmp_path):
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=3)
    nli = _make_nli(0.8)
    _fit(t, _make_llm(3), nli)

    path = tmp_path / "model.json"
    t.save(path)

    nli2 = _make_nli(0.8)
    with patch("semaxis.unsupervised.NLIModel", return_value=nli2):
        loaded = UnsupervisedTransformer.load(path, llm=MagicMock())

    assert loaded.features_ == t.features_
    assert loaded.nli_model == t.nli_model
    texts = ["text a", "text b"]
    np.testing.assert_array_equal(t.transform(texts), loaded.transform(texts))


def test_save_load_restores_nli_model_name(tmp_path):
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="custom-nli", n_features=2)
    nli = _make_nli()
    _fit(t, _make_llm(2), nli)

    path = tmp_path / "model.json"
    t.save(path)

    with patch("semaxis.unsupervised.NLIModel") as MockNLI:
        MockNLI.return_value = _make_nli()
        UnsupervisedTransformer.load(path, llm=MagicMock())
    MockNLI.assert_called_once_with("custom-nli")


def test_transform_empty_features_raises():
    t = UnsupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=0)
    nli = _make_nli()
    _fit(t, _make_llm(0), nli)
    with pytest.raises(ValueError, match="No features"):
        t.transform(["text"])
