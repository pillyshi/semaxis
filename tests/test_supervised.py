"""Unit tests for SupervisedTransformer."""
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from semaxis import SupervisedTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm(n_features: int = 2) -> MagicMock:
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


def _fit_binary(
    transformer: SupervisedTransformer,
    llm: MagicMock,
    nli: MagicMock,
) -> SupervisedTransformer:
    texts = ["text A1", "text A2", "text B1", "text B2"]
    labels = [0, 0, 1, 1]
    with patch("semaxis.supervised.NLIModel", return_value=nli):
        transformer.llm = llm
        return transformer.fit(texts, labels)


# ---------------------------------------------------------------------------
# fit — binary
# ---------------------------------------------------------------------------

def test_fit_binary_returns_self():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    result = _fit_binary(t, _make_llm(2), _make_nli())
    assert result is t


def test_fit_binary_sets_classes():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    _fit_binary(t, _make_llm(2), _make_nli())
    np.testing.assert_array_equal(t.classes_, [0, 1])


def test_fit_binary_sets_features():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    _fit_binary(t, _make_llm(2), _make_nli())
    assert t.features_ == ["hyp 0", "hyp 1"]


def test_fit_binary_sets_feature_meta_parallel():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    _fit_binary(t, _make_llm(2), _make_nli())
    assert len(t.feature_meta_) == len(t.features_)


def test_fit_binary_meta_labels():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    _fit_binary(t, _make_llm(2), _make_nli())
    for meta in t.feature_meta_:
        assert meta.positive == 0
        assert meta.negative == 1


def test_fit_binary_string_labels():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    llm = _make_llm(2)
    nli = _make_nli()
    texts = ["text A", "text B", "text C", "text D"]
    labels = ["cat", "cat", "dog", "dog"]
    with patch("semaxis.supervised.NLIModel", return_value=nli):
        t.llm = llm
        t.fit(texts, labels)
    np.testing.assert_array_equal(t.classes_, ["cat", "dog"])
    for meta in t.feature_meta_:
        assert meta.positive == "cat"
        assert meta.negative == "dog"


def test_fit_binary_ignores_strategy():
    """Binary: ovr and ovo produce the same single pair."""
    texts = ["a", "b", "c", "d"]
    labels = [0, 0, 1, 1]
    llm = _make_llm(2)
    nli = _make_nli()

    t_ovr = SupervisedTransformer(llm=llm, nli_model="m", n_features=2, strategy="ovr")
    t_ovo = SupervisedTransformer(llm=llm, nli_model="m", n_features=2, strategy="ovo")

    with patch("semaxis.supervised.NLIModel", return_value=nli):
        t_ovr.fit(texts, labels)
        t_ovo.fit(texts, labels)

    assert len(t_ovr.features_) == len(t_ovo.features_)
    assert t_ovr.feature_meta_ == t_ovo.feature_meta_


# ---------------------------------------------------------------------------
# fit — multi-class OvR
# ---------------------------------------------------------------------------

def test_fit_ovr_feature_count():
    """OvR: n_classes pairs × n_features hypotheses."""
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2, strategy="ovr")
    llm = _make_llm(2)
    nli = _make_nli()
    texts = ["a", "b", "c", "d", "e", "f"]
    labels = [0, 0, 1, 1, 2, 2]
    with patch("semaxis.supervised.NLIModel", return_value=nli):
        t.llm = llm
        t.fit(texts, labels)
    assert len(t.features_) == 3 * 2  # 3 classes × 2 features


def test_fit_ovr_meta_has_rest():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2, strategy="ovr")
    llm = _make_llm(2)
    nli = _make_nli()
    texts = ["a", "b", "c", "d", "e", "f"]
    labels = [0, 0, 1, 1, 2, 2]
    with patch("semaxis.supervised.NLIModel", return_value=nli):
        t.llm = llm
        t.fit(texts, labels)
    assert all(meta.negative == "rest" for meta in t.feature_meta_)


def test_fit_ovr_meta_covers_all_classes():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2, strategy="ovr")
    llm = _make_llm(2)
    nli = _make_nli()
    texts = ["a", "b", "c", "d", "e", "f"]
    labels = [0, 0, 1, 1, 2, 2]
    with patch("semaxis.supervised.NLIModel", return_value=nli):
        t.llm = llm
        t.fit(texts, labels)
    positives = {meta.positive for meta in t.feature_meta_}
    assert positives == {0, 1, 2}


# ---------------------------------------------------------------------------
# fit — multi-class OvO
# ---------------------------------------------------------------------------

def test_fit_ovo_feature_count():
    """OvO: C(n_classes, 2) pairs × n_features hypotheses."""
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2, strategy="ovo")
    llm = _make_llm(2)
    nli = _make_nli()
    texts = ["a", "b", "c", "d", "e", "f"]
    labels = [0, 0, 1, 1, 2, 2]
    with patch("semaxis.supervised.NLIModel", return_value=nli):
        t.llm = llm
        t.fit(texts, labels)
    assert len(t.features_) == 3 * 2  # C(3,2)=3 pairs × 2 features


def test_fit_ovo_meta_pairs():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2, strategy="ovo")
    llm = _make_llm(2)
    nli = _make_nli()
    texts = ["a", "b", "c", "d", "e", "f"]
    labels = [0, 0, 1, 1, 2, 2]
    with patch("semaxis.supervised.NLIModel", return_value=nli):
        t.llm = llm
        t.fit(texts, labels)
    pairs = {(meta.positive, meta.negative) for meta in t.feature_meta_}
    assert pairs == {(0, 1), (0, 2), (1, 2)}


# ---------------------------------------------------------------------------
# fit — validation
# ---------------------------------------------------------------------------

def test_fit_invalid_strategy_raises():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", strategy="invalid")
    with pytest.raises(ValueError, match="strategy"):
        t.fit(["a", "b"], [0, 1])


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------

def test_transform_shape():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=3)
    nli = _make_nli(0.5)
    _fit_binary(t, _make_llm(3), nli)

    test_texts = ["x", "y", "z", "w"]
    X = t.transform(test_texts)
    assert X.shape == (4, 3)


def test_transform_values():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    nli = _make_nli(0.7)
    _fit_binary(t, _make_llm(2), nli)

    X = t.transform(["x", "y"])
    np.testing.assert_allclose(X, 0.7)


def test_transform_uses_fitted_features():
    """NLI is called once per hypothesis in features_."""
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=3)
    nli = _make_nli()
    _fit_binary(t, _make_llm(3), nli)

    t.transform(["x", "y"])
    assert nli.score.call_count == 3


# ---------------------------------------------------------------------------
# sample_method parameter
# ---------------------------------------------------------------------------

def test_sample_method_invalid_raises():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", sample_method="bad")
    with pytest.raises(ValueError, match="sample_method"):
        _fit_binary(t, _make_llm(2), _make_nli())


def test_sample_method_random_default():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    assert t.sample_method == "random"


def _fit_binary_with_method(method: str) -> SupervisedTransformer:
    t = SupervisedTransformer(
        llm=MagicMock(), nli_model="m", n_features=2, sample_method=method
    )
    texts = ["text A1", "text A2", "text B1", "text B2"]
    labels = [0, 0, 1, 1]
    nli = _make_nli()
    llm = _make_llm(2)
    fake_embeddings = np.zeros((2, 4))
    fake_model = MagicMock()
    fake_model.encode.return_value = fake_embeddings
    with patch("semaxis.supervised.NLIModel", return_value=nli), \
         patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        t.llm = llm
        t.fit(texts, labels)
    return t


def test_sample_method_kmeans_fits_without_error():
    t = _fit_binary_with_method("kmeans")
    assert t.features_ == ["hyp 0", "hyp 1"]


def test_sample_method_votek_fits_without_error():
    t = _fit_binary_with_method("votek")
    assert t.features_ == ["hyp 0", "hyp 1"]


# ---------------------------------------------------------------------------
# sklearn clone compatibility
# ---------------------------------------------------------------------------

def test_sklearn_clone_with_llm_client_instance():
    """clone() must not raise even when llm holds a non-picklable object."""
    llm = MagicMock()
    llm._lock = threading.RLock()  # simulate unpicklable state
    t = SupervisedTransformer(llm=llm, nli_model="m", n_features=2)
    cloned = clone(t)
    assert cloned is not t
    assert cloned.llm is llm


def test_sklearn_clone_preserves_params():
    llm = MagicMock()
    t = SupervisedTransformer(
        llm=llm, nli_model="my-nli", n_features=5, strategy="ovo",
        context_limit=50_000, language="en", seed=42,
        sample_method="kmeans", embedding_model="my-embed",
    )
    cloned = clone(t)
    assert cloned.get_params(deep=False) == t.get_params(deep=False)


def test_sklearn_clone_with_string_llm():
    t = SupervisedTransformer(llm="gpt-4o", nli_model="m")
    cloned = clone(t)
    assert cloned.llm == "gpt-4o"


def test_sklearn_clone_preserves_sklearn_output_config():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m")
    t._sklearn_output_config = {"transform": "pandas"}
    cloned = clone(t)
    assert cloned._sklearn_output_config == {"transform": "pandas"}


def test_sklearn_clone_preserves_metadata_request():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m")
    t._metadata_request = {"key": "value"}
    cloned = clone(t)
    assert cloned._metadata_request == {"key": "value"}
    assert cloned._metadata_request is not t._metadata_request  # deep-copied


def test_sample_method_kmeans_calls_sentence_transformer():
    texts = ["text A1", "text A2", "text B1", "text B2"]
    labels = [0, 0, 1, 1]
    nli = _make_nli()
    llm = _make_llm(2)
    fake_embeddings = np.zeros((2, 4))
    fake_model = MagicMock()
    fake_model.encode.return_value = fake_embeddings

    t = SupervisedTransformer(
        llm=MagicMock(), nli_model="m", n_features=2, sample_method="kmeans"
    )
    with patch("semaxis.supervised.NLIModel", return_value=nli), \
         patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as MockST:
        t.llm = llm
        t.fit(texts, labels)
        assert MockST.called


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def test_save_before_fit_raises(tmp_path):
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m")
    with pytest.raises(NotFittedError):
        t.save(tmp_path / "model.json")


def test_save_load_roundtrip(tmp_path):
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    nli = _make_nli(0.7)
    _fit_binary(t, _make_llm(2), nli)

    path = tmp_path / "model.json"
    t.save(path)

    nli2 = _make_nli(0.7)
    with patch("semaxis.supervised.NLIModel", return_value=nli2):
        loaded = SupervisedTransformer.load(path, llm=MagicMock())

    assert loaded.features_ == t.features_
    assert loaded.nli_model == t.nli_model
    np.testing.assert_array_equal(loaded.classes_, t.classes_)
    assert loaded.classes_.dtype == t.classes_.dtype
    assert loaded.feature_meta_ == t.feature_meta_
    texts = ["text a", "text b"]
    np.testing.assert_array_equal(t.transform(texts), loaded.transform(texts))


def test_save_load_restores_nli_model_name(tmp_path):
    t = SupervisedTransformer(llm=MagicMock(), nli_model="custom-nli", n_features=2)
    nli = _make_nli()
    _fit_binary(t, _make_llm(2), nli)

    path = tmp_path / "model.json"
    t.save(path)

    with patch("semaxis.supervised.NLIModel") as MockNLI:
        MockNLI.return_value = _make_nli()
        SupervisedTransformer.load(path, llm=MagicMock())
    MockNLI.assert_called_once_with("custom-nli")


def test_save_load_preserves_classes_dtype(tmp_path):
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    nli = _make_nli()
    _fit_binary(t, _make_llm(2), nli)

    path = tmp_path / "model.json"
    t.save(path)

    with patch("semaxis.supervised.NLIModel", return_value=_make_nli()):
        loaded = SupervisedTransformer.load(path, llm=MagicMock())
    assert loaded.classes_.dtype == t.classes_.dtype


def test_transform_before_fit_raises_not_fitted_error():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m")
    with pytest.raises(NotFittedError):
        t.transform(["text"])


def test_transform_empty_features_raises():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=0)
    llm = _make_llm(0)
    nli = _make_nli()
    _fit_binary(t, llm, nli)
    with pytest.raises(ValueError, match="No features"):
        t.transform(["text"])


# ---------------------------------------------------------------------------
# fit — multi-label
# ---------------------------------------------------------------------------

def _fit_multilabel(
    transformer: SupervisedTransformer,
    llm: MagicMock,
    nli: MagicMock,
) -> SupervisedTransformer:
    texts = ["t0", "t1", "t2", "t3", "t4", "t5"]
    # 3 samples per label combination; 2 labels
    y = np.array([
        [1, 0],
        [1, 1],
        [0, 1],
        [1, 0],
        [0, 0],
        [0, 1],
    ])
    with patch("semaxis.supervised.NLIModel", return_value=nli):
        transformer.llm = llm
        return transformer.fit(texts, y)


def test_fit_multilabel_returns_self():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    result = _fit_multilabel(t, _make_llm(2), _make_nli())
    assert result is t


def test_fit_multilabel_sets_classes():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    _fit_multilabel(t, _make_llm(2), _make_nli())
    np.testing.assert_array_equal(t.classes_, [0, 1])


def test_fit_multilabel_feature_count():
    """Multi-label: n_labels × n_features hypotheses."""
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    _fit_multilabel(t, _make_llm(2), _make_nli())
    assert len(t.features_) == 2 * 2  # 2 labels × 2 features


def test_fit_multilabel_meta_all_rest():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    _fit_multilabel(t, _make_llm(2), _make_nli())
    assert all(meta.negative == "rest" for meta in t.feature_meta_)


def test_fit_multilabel_meta_covers_all_labels():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    _fit_multilabel(t, _make_llm(2), _make_nli())
    positives = {meta.positive for meta in t.feature_meta_}
    assert positives == {0, 1}


def test_fit_multilabel_ovo_raises():
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", strategy="ovo")
    y = np.array([[1, 0], [0, 1], [1, 1]])
    with pytest.raises(ValueError, match="multi-label"):
        with patch("semaxis.supervised.NLIModel", return_value=_make_nli()):
            t.llm = _make_llm(2)
            t.fit(["a", "b", "c"], y)


def test_save_load_roundtrip_multilabel(tmp_path):
    t = SupervisedTransformer(llm=MagicMock(), nli_model="m", n_features=2)
    nli = _make_nli(0.6)
    _fit_multilabel(t, _make_llm(2), nli)

    path = tmp_path / "model_ml.json"
    t.save(path)

    nli2 = _make_nli(0.6)
    with patch("semaxis.supervised.NLIModel", return_value=nli2):
        loaded = SupervisedTransformer.load(path, llm=MagicMock())

    assert loaded.features_ == t.features_
    np.testing.assert_array_equal(loaded.classes_, t.classes_)
    assert loaded.feature_meta_ == t.feature_meta_
