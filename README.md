# SemAxis

**Interpretable NLI-based text features for scikit-learn.**

SemAxis turns raw text into a feature matrix by asking an LLM to generate natural-language hypotheses and scoring each text against them with an NLI model. Every feature is a human-readable sentence — no black-box embeddings.

```
texts  ──►  LLM (hypothesis generation)  ──►  NLI scoring  ──►  X: (n_texts, n_features)
```

Both transformers are sklearn-compatible: they work inside `Pipeline` and are safe to use with `cross_val_score`.

---

## Installation

```bash
pip install semaxis
```

---

## Unsupervised

`UnsupervisedTransformer` generates hypotheses that characterize the text collection without labels.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from semaxis import UnsupervisedTransformer

pipe = Pipeline([
    ("vect", UnsupervisedTransformer(
        llm="gpt-4o",
        nli_model="cross-encoder/nli-deberta-v3-large",
        n_features=20,
    )),
    ("clf", LogisticRegression()),
])

cross_val_score(pipe, texts, labels, cv=5)
```

---

## Supervised

`SupervisedTransformer` generates hypotheses that *discriminate* between classes.

### Binary

```python
from semaxis import SupervisedTransformer

pipe = Pipeline([
    ("vect", SupervisedTransformer(
        llm="gpt-4o",
        nli_model="cross-encoder/nli-deberta-v3-large",
        n_features=20,
    )),
    ("clf", LogisticRegression()),
])

cross_val_score(pipe, texts, labels, cv=5)
```

Labels can be numeric (`0`/`1`) or strings (`"positive"`/`"negative"`).

### Multi-class

Use `strategy="ovr"` (one-vs-rest, default) or `strategy="ovo"` (one-vs-one):

```python
# OvR: generates n_features hypotheses per class (k × n_features total)
vect = SupervisedTransformer(llm="gpt-4o", nli_model="...", n_features=10, strategy="ovr")

# OvO: generates n_features hypotheses per class pair (C(k,2) × n_features total)
vect = SupervisedTransformer(llm="gpt-4o", nli_model="...", n_features=10, strategy="ovo")
```

---

## Interpreting features

After fitting, both transformers expose `features_` (flat list of hypothesis strings).
`SupervisedTransformer` also exposes `feature_meta_`, parallel to `features_`, which records which class pair each hypothesis came from.

```python
vect = SupervisedTransformer(llm="gpt-4o", nli_model="...", n_features=5, strategy="ovr")
vect.fit(texts, labels)

for hypothesis, meta in zip(vect.features_, vect.feature_meta_):
    print(f"[{meta.positive} vs {meta.negative}]  {hypothesis}")
```

```
[cat vs rest]  This text describes feline behavior.
[cat vs rest]  This text mentions a cat or kitten.
[dog vs rest]  This text describes canine behavior.
[dog vs rest]  This text mentions a dog or puppy.
...
```

Combine with a linear model to get per-hypothesis coefficients:

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

X = vect.transform(texts)
clf = LogisticRegression().fit(X, labels)

for coef, hyp in sorted(zip(clf.coef_[0], vect.features_), key=lambda x: abs(x[0]), reverse=True):
    print(f"  {coef:+.3f}  {hyp}")
```

---

## Hard Positive Oversampling

`HardPositiveOverSampler` generates synthetic positive-class texts that are semantically valid but superficially ambiguous — texts that experts would label positive but shallow classifiers might not.

It implements the imbalanced-learn `fit_resample(X, y)` interface for raw text arrays. Only binary labels (`0`/`1`) are supported.

```python
from semaxis import HardPositiveOverSampler

sampler = HardPositiveOverSampler(llm="gpt-4o", n_synthesized=20)
X_aug, y_aug = sampler.fit_resample(texts, labels)
```

By default (`n_synthesized=None`) it generates enough samples to balance the classes.

### Inspecting generated samples

After `fit_resample`, `generation_result_` exposes the full LLM analysis:

```python
for hp in sampler.generation_result_.hard_positives:
    print(hp.text)
    print("  evidence:", hp.positive_evidence)
    print("  confusing:", hp.confusing_evidence)
```

### Saving and loading

```python
sampler.save("hard_positives.json")

sampler2 = HardPositiveOverSampler.load("hard_positives.json", llm="gpt-4o")
```

### Logging progress

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("semaxis")

sampler = HardPositiveOverSampler(
    llm="gpt-4o",
    n_synthesized=20,
    verbose=True,   # tqdm progress bar
    logger=logger,  # per-batch debug logging
)
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_synthesized` | `None` | Number of hard positives to generate. `None` balances the classes. |
| `batch_size` | `3` | Samples requested per LLM call. |
| `language` | `None` | Constrain generated texts to a specific language. |
| `deduplicate` | `True` | Reject generated texts that duplicate existing ones. |
| `seed` | `None` | Random seed for reproducible example sampling. |

---

## Custom LLM

For in-process inference, use `LlamaCppClient` backed by [llama-cpp-python](https://github.com/abetlen/llama-cpp-python):

```python
from llama_cpp import Llama
from semaxis import UnsupervisedTransformer
from semaxis import LlamaCppClient

llm = LlamaCppClient(Llama(model_path="path/to/model.gguf", n_ctx=4096))

vect = UnsupervisedTransformer(llm=llm, nli_model="cross-encoder/nli-deberta-v3-large")
```

Install the optional dependency with:

```bash
pip install "semaxis[llamacpp]"
```

---

## Related Work

- Balek et al. (2025) — [LLM-based feature generation for interpretable ML](https://arxiv.org/abs/2409.07132)
- Yin et al. (2019) — NLI as zero-shot text classifier
- LogiPart (2025) — LLM hypothesis generation + NLI propagation

---

## Citation

```bibtex
@software{semaxis2026,
  title  = {SemAxis: Interpretable NLI-based text features for scikit-learn},
  year   = {2026},
}
```
