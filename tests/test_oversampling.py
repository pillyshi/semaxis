"""Unit tests for HardPositiveOverSampler."""
import threading
from unittest.mock import MagicMock, patch

import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from semaxis import (
    BoundaryFeature,
    HardPositive,
    HardPositiveGenerationResult,
    HardPositiveOverSampler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm(n_hard_positives: int = 1) -> MagicMock:
    llm = MagicMock()
    llm.count_tokens.return_value = 1
    llm.complete_structured.return_value = HardPositiveGenerationResult(
        positive_features=["pf1"],
        negative_features=["nf1"],
        boundary_features=[BoundaryFeature(feature="bf1", importance=0.9)],
        hard_positives=[
            HardPositive(
                text=f"gen {i}",
                positive_evidence=["evidence"],
                confusing_evidence=["confusing"],
            )
            for i in range(n_hard_positives)
        ],
    )
    return llm


def _fit_resample(
    sampler: HardPositiveOverSampler,
    llm: MagicMock,
) -> tuple[list[str], list[int]]:
    sampler.llm = llm
    texts = ["pos A", "pos B", "neg C", "neg D"]
    labels = [1, 1, 0, 0]
    return sampler.fit_resample(texts, labels)


# ---------------------------------------------------------------------------
# fit_resample — return values
# ---------------------------------------------------------------------------

def test_fit_resample_returns_augmented_texts():
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=1)
    llm = _make_llm(n_hard_positives=1)
    X_aug, _ = _fit_resample(sampler, llm)
    assert "gen 0" in X_aug


def test_fit_resample_preserves_original_texts():
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=1)
    llm = _make_llm(n_hard_positives=1)
    X_aug, _ = _fit_resample(sampler, llm)
    assert "pos A" in X_aug
    assert "neg C" in X_aug


def test_fit_resample_appended_labels_are_positive():
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=2)
    llm = _make_llm(n_hard_positives=2)
    _, y_aug = _fit_resample(sampler, llm)
    assert y_aug[-2:] == [1, 1]


def test_fit_resample_length():
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=3)
    llm = _make_llm(n_hard_positives=3)
    X_aug, y_aug = _fit_resample(sampler, llm)
    assert len(X_aug) == len(y_aug) == 4 + 3


# ---------------------------------------------------------------------------
# fit_resample — generation_result_ attribute
# ---------------------------------------------------------------------------

def test_fit_resample_sets_generation_result():
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=1)
    llm = _make_llm(n_hard_positives=1)
    _fit_resample(sampler, llm)
    assert isinstance(sampler.generation_result_, HardPositiveGenerationResult)


def test_generation_result_hard_positives_count():
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=2)
    llm = _make_llm(n_hard_positives=2)
    _fit_resample(sampler, llm)
    assert len(sampler.generation_result_.hard_positives) == 2


def test_generation_result_has_boundary_features():
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=1)
    llm = _make_llm(n_hard_positives=1)
    _fit_resample(sampler, llm)
    assert len(sampler.generation_result_.boundary_features) == 1
    assert isinstance(sampler.generation_result_.boundary_features[0], BoundaryFeature)


def test_generation_result_hard_positive_type():
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=1)
    llm = _make_llm(n_hard_positives=1)
    _fit_resample(sampler, llm)
    hp = sampler.generation_result_.hard_positives[0]
    assert isinstance(hp, HardPositive)
    assert hp.text == "gen 0"


# ---------------------------------------------------------------------------
# sample_method validation
# ---------------------------------------------------------------------------

def test_sample_method_invalid_raises():
    sampler = HardPositiveOverSampler(llm=MagicMock(), sample_method="bad")
    with pytest.raises(ValueError, match="sample_method"):
        sampler.fit_resample(["a", "b"], [1, 0])


def test_fit_resample_mismatched_lengths_raises():
    sampler = HardPositiveOverSampler(llm=MagicMock())
    with pytest.raises(ValueError, match="same length"):
        sampler.fit_resample(["a", "b", "c"], [1, 0])


def test_fit_resample_non_binary_labels_raises():
    sampler = HardPositiveOverSampler(llm=MagicMock())
    with pytest.raises(ValueError, match="binary"):
        sampler.fit_resample(["a", "b", "c"], [0, 1, 2])


def test_fit_resample_string_labels_raises():
    sampler = HardPositiveOverSampler(llm=MagicMock())
    with pytest.raises(ValueError, match="binary"):
        sampler.fit_resample(["a", "b"], ["pos", "neg"])


def test_fit_resample_all_positive_raises():
    sampler = HardPositiveOverSampler(llm=MagicMock())
    with pytest.raises(ValueError, match="negative"):
        sampler.fit_resample(["a", "b"], [1, 1])


def test_fit_resample_all_negative_raises():
    sampler = HardPositiveOverSampler(llm=MagicMock())
    with pytest.raises(ValueError, match="positive"):
        sampler.fit_resample(["a", "b"], [0, 0])


def test_fit_resample_context_limit_too_small_raises():
    sampler = HardPositiveOverSampler(llm=MagicMock(), context_limit=400)
    with pytest.raises(ValueError, match="context_limit"):
        sampler.fit_resample(["a", "b"], [1, 0])


def test_fit_resample_context_limit_zero_budget_raises():
    # context_limit=501 passes the old <= 500 guard but yields budget = (501-500)//2 = 0
    sampler = HardPositiveOverSampler(llm=MagicMock(), context_limit=501)
    with pytest.raises(ValueError, match="context_limit"):
        sampler.fit_resample(["a", "b"], [1, 0])


def test_fit_resample_n_synthesized_zero_returns_original():
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=0)
    llm = MagicMock()
    sampler.llm = llm
    X_aug, y_aug = sampler.fit_resample(["a", "b"], [1, 0])
    assert X_aug == ["a", "b"]
    assert y_aug == [1, 0]
    llm.complete_structured.assert_not_called()


def test_fit_resample_n_synthesized_negative_raises():
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=-5)
    with pytest.raises(ValueError, match="n_synthesized"):
        sampler.fit_resample(["a", "b"], [1, 0])


def test_fit_resample_batch_size_invalid_raises():
    sampler = HardPositiveOverSampler(llm=MagicMock(), batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        sampler.fit_resample(["a", "b"], [1, 0])


def test_fit_resample_max_examples_per_class_invalid_raises():
    sampler = HardPositiveOverSampler(llm=MagicMock(), max_examples_per_class=0)
    with pytest.raises(ValueError, match="max_examples_per_class"):
        sampler.fit_resample(["a", "b"], [1, 0])


def test_fit_resample_skips_batch_on_exception_and_warns():
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=1)
    llm = MagicMock()
    llm.count_tokens.return_value = 1
    llm.complete_structured.side_effect = ValueError("LLM returned an unexpected JSON structure")
    sampler.llm = llm
    with pytest.warns(UserWarning, match="Skipping batch"):
        X_aug, y_aug = sampler.fit_resample(["pos A", "neg B"], [1, 0])
    assert X_aug == ["pos A", "neg B"]
    assert y_aug == [1, 0]
    assert sampler.generation_result_.hard_positives == []


def test_fit_resample_count_mismatch_warns():
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=3)
    llm = _make_llm(n_hard_positives=1)  # LLM returns 1, but 3 requested
    sampler.llm = llm
    with pytest.warns(UserWarning, match="1 accepted hard positives, expected 3"):
        sampler.fit_resample(["pos A", "pos B", "neg C", "neg D"], [1, 1, 0, 0])


def test_fit_resample_pos_sampled_empty_warns():
    # budget = (context_limit - 500) // 2 = (2000 - 500) // 2 = 750
    # count_tokens returns 1000 > 750, so every text exceeds budget → pos_sampled=[]
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=1, context_limit=2_000)
    llm = _make_llm(n_hard_positives=1)
    llm.count_tokens.return_value = 1_000
    sampler.llm = llm
    with pytest.warns(UserWarning, match="positive texts exceed"):
        sampler.fit_resample(["pos A", "pos B", "neg C", "neg D"], [1, 1, 0, 0])


def test_fit_resample_neg_sampled_empty_warns():
    # Same budget setup; negative texts also all exceed budget → neg_sampled=[]
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=1, context_limit=2_000)
    llm = _make_llm(n_hard_positives=1)
    llm.count_tokens.return_value = 1_000
    sampler.llm = llm
    with pytest.warns(UserWarning, match="negative texts exceed"):
        sampler.fit_resample(["pos A", "pos B", "neg C", "neg D"], [1, 1, 0, 0])


def test_sample_method_random_default():
    sampler = HardPositiveOverSampler(llm=MagicMock())
    assert sampler.sample_method == "random"


def test_language_default_none():
    sampler = HardPositiveOverSampler(llm=MagicMock())
    assert sampler.language is None


def test_language_instruction_applies_only_to_generated_text():
    sampler = HardPositiveOverSampler(
        llm=MagicMock(),
        n_synthesized=1,
        language="Japanese",
    )
    llm = _make_llm(n_hard_positives=1)
    _fit_resample(sampler, llm)

    messages = llm.complete_structured.call_args.args[0]
    user_message = messages[1]["content"]
    assert "Write only hard_positives[].text in Japanese" in user_message
    assert "Do not force positive_features" in user_message
    assert "confusing_evidence to be written in Japanese" in user_message


def test_default_n_synthesized_balances_positive_class():
    sampler = HardPositiveOverSampler(llm=MagicMock())
    llm = _make_llm(n_hard_positives=1)
    sampler.llm = llm
    X_aug, y_aug = sampler.fit_resample(
        ["pos A", "neg B", "neg C"],
        [1, 0, 0],
    )
    assert X_aug[-1:] == ["gen 0"]
    assert y_aug == [1, 0, 0, 1]


def test_default_n_synthesized_noops_when_positive_not_smaller():
    sampler = HardPositiveOverSampler(llm=MagicMock())
    llm = MagicMock()
    sampler.llm = llm
    X_aug, y_aug = sampler.fit_resample(
        ["pos A", "pos B", "neg C"],
        [1, 1, 0],
    )
    assert X_aug == ["pos A", "pos B", "neg C"]
    assert y_aug == [1, 1, 0]
    llm.complete_structured.assert_not_called()


def test_fit_resample_generates_iterative_batches():
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=5, batch_size=2)
    llm = MagicMock()
    llm.count_tokens.return_value = 1
    llm.complete_structured.side_effect = [
        HardPositiveGenerationResult(
            positive_features=["pf1"],
            negative_features=["nf1"],
            boundary_features=[BoundaryFeature(feature="bf1", importance=0.9)],
            hard_positives=[
                HardPositive(text="gen 0", positive_evidence=[], confusing_evidence=[]),
                HardPositive(text="gen 1", positive_evidence=[], confusing_evidence=[]),
            ],
        ),
        HardPositiveGenerationResult(
            positive_features=["pf2"],
            negative_features=["nf2"],
            boundary_features=[BoundaryFeature(feature="bf2", importance=0.8)],
            hard_positives=[
                HardPositive(text="gen 2", positive_evidence=[], confusing_evidence=[]),
                HardPositive(text="gen 3", positive_evidence=[], confusing_evidence=[]),
            ],
        ),
        HardPositiveGenerationResult(
            positive_features=["pf3"],
            negative_features=["nf3"],
            boundary_features=[BoundaryFeature(feature="bf3", importance=0.7)],
            hard_positives=[
                HardPositive(text="gen 4", positive_evidence=[], confusing_evidence=[]),
            ],
        ),
    ]
    sampler.llm = llm

    X_aug, _ = sampler.fit_resample(["pos A", "neg B"], [1, 0])

    assert X_aug[-5:] == ["gen 0", "gen 1", "gen 2", "gen 3", "gen 4"]
    assert llm.complete_structured.call_count == 3
    requested_counts = [
        call.args[0][1]["content"].split("Count: ")[1]
        for call in llm.complete_structured.call_args_list
    ]
    assert requested_counts == ["2", "2", "1"]
    assert sampler.generation_result_.positive_features == ["pf1", "pf2", "pf3"]


def test_fit_resample_continues_after_short_llm_responses():
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=3, batch_size=2)
    llm = MagicMock()
    llm.count_tokens.return_value = 1
    llm.complete_structured.side_effect = [
        HardPositiveGenerationResult(
            positive_features=[],
            negative_features=[],
            boundary_features=[],
            hard_positives=[
                HardPositive(text=f"gen {i}", positive_evidence=[], confusing_evidence=[])
            ],
        )
        for i in range(3)
    ]
    sampler.llm = llm

    X_aug, _ = sampler.fit_resample(["pos A", "neg B"], [1, 0])

    assert X_aug[-3:] == ["gen 0", "gen 1", "gen 2"]
    assert llm.complete_structured.call_count == 3


def test_fit_resample_deduplicates_exact_texts_by_default():
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=2, batch_size=2)
    llm = MagicMock()
    llm.count_tokens.return_value = 1
    llm.complete_structured.side_effect = [
        HardPositiveGenerationResult(
            positive_features=[],
            negative_features=[],
            boundary_features=[],
            hard_positives=[
                HardPositive(text="pos A", positive_evidence=[], confusing_evidence=[]),
                HardPositive(text="gen 0", positive_evidence=[], confusing_evidence=[]),
            ],
        ),
        HardPositiveGenerationResult(
            positive_features=[],
            negative_features=[],
            boundary_features=[],
            hard_positives=[
                HardPositive(text="gen 0", positive_evidence=[], confusing_evidence=[]),
                HardPositive(text="gen 1", positive_evidence=[], confusing_evidence=[]),
            ],
        ),
    ]
    sampler.llm = llm

    X_aug, _ = sampler.fit_resample(["pos A", "neg B"], [1, 0])

    assert X_aug == ["pos A", "neg B", "gen 0", "gen 1"]


def test_fit_resample_deduplicate_false_preserves_exact_duplicates():
    sampler = HardPositiveOverSampler(
        llm=MagicMock(),
        n_synthesized=2,
        batch_size=2,
        deduplicate=False,
    )
    llm = _make_llm(n_hard_positives=2)
    llm.complete_structured.return_value.hard_positives[1].text = "gen 0"
    sampler.llm = llm

    X_aug, _ = sampler.fit_resample(["pos A", "neg B"], [1, 0])

    assert X_aug == ["pos A", "neg B", "gen 0", "gen 0"]


def test_fit_resample_max_examples_per_class_caps_prompt_examples():
    sampler = HardPositiveOverSampler(
        llm=MagicMock(),
        n_synthesized=1,
        max_examples_per_class=1,
    )
    llm = _make_llm(n_hard_positives=1)
    sampler.llm = llm

    sampler.fit_resample(["pos A", "pos B", "neg C", "neg D"], [1, 1, 0, 0])

    user_message = llm.complete_structured.call_args.args[0][1]["content"]
    assert user_message.count("\n2. ") == 0


def _fit_resample_with_method(method: str) -> tuple[list[str], list[int]]:
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=1, sample_method=method)
    texts = ["pos A", "pos B", "neg C", "neg D"]
    labels = [1, 1, 0, 0]
    llm = _make_llm(n_hard_positives=1)
    fake_embeddings = __import__("numpy").zeros((2, 4))
    fake_model = MagicMock()
    fake_model.encode.return_value = fake_embeddings
    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        sampler.llm = llm
        return sampler.fit_resample(texts, labels)


def test_sample_method_kmeans_fits_without_error():
    X_aug, _ = _fit_resample_with_method("kmeans")
    assert "gen 0" in X_aug


def test_sample_method_votek_fits_without_error():
    X_aug, _ = _fit_resample_with_method("votek")
    assert "gen 0" in X_aug


# ---------------------------------------------------------------------------
# sklearn clone compatibility
# ---------------------------------------------------------------------------

def test_sklearn_clone_with_llm_client_instance():
    llm = MagicMock()
    llm._lock = threading.RLock()
    sampler = HardPositiveOverSampler(llm=llm, n_synthesized=5)
    cloned = clone(sampler)
    assert cloned is not sampler
    assert cloned.llm is llm


def test_sklearn_clone_preserves_params():
    llm = MagicMock()
    sampler = HardPositiveOverSampler(
        llm=llm,
        n_synthesized=20,
        context_limit=50_000,
        seed=42,
        sample_method="kmeans",
        embedding_model="my-embed",
    )
    cloned = clone(sampler)
    assert cloned.get_params(deep=False) == sampler.get_params(deep=False)


def test_sklearn_clone_with_string_llm():
    sampler = HardPositiveOverSampler(llm="gpt-4o")
    cloned = clone(sampler)
    assert cloned.llm == "gpt-4o"


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def test_save_before_fit_raises(tmp_path):
    sampler = HardPositiveOverSampler(llm=MagicMock())
    with pytest.raises(NotFittedError):
        sampler.save(tmp_path / "sampler.json")


def test_save_load_roundtrip(tmp_path):
    sampler = HardPositiveOverSampler(llm=MagicMock(), n_synthesized=1)
    llm = _make_llm(n_hard_positives=1)
    _fit_resample(sampler, llm)

    path = tmp_path / "sampler.json"
    sampler.save(path)

    loaded = HardPositiveOverSampler.load(path, llm=MagicMock())
    assert loaded.generation_result_ == sampler.generation_result_
