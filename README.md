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

## Custom LLM

Any [LangChain](https://python.langchain.com/)-compatible model works via `LangChainLLMClient`:

```python
from langchain_ollama import ChatOllama
from semaxis import UnsupervisedTransformer
from semaxis import LangChainLLMClient

llm = LangChainLLMClient(ChatOllama(model="llama3.2", format="json"))

vect = UnsupervisedTransformer(llm=llm, nli_model="cross-encoder/nli-deberta-v3-large")
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
