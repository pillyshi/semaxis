# Balek et al. (2025)

## Citation

Vojtech Balek, Lukas Sykora, Vilem Sklenak, and Tomas Kliegr. 2025.
*LLM-based feature generation from text for interpretable machine learning*.
Machine Learning. DOI:
[`10.1007/s10994-025-06867-1`](https://doi.org/10.1007/s10994-025-06867-1).

- OpenAlex journal record: `W4414681764`
- Public preprint record: `W4403621451`
- Public preprint DOI: `10.48550/arXiv.2409.07132`
- Local PDF cache: `../pdfs/balek-2024-preprint.pdf`
- Public full text used for reading: <https://arxiv.org/abs/2409.07132>

OpenAlex identifies the 2025 *Machine Learning* article as the final journal
record but does not expose open full text for it. This note cites that journal
publication and uses the public arXiv version for content inspection.

## Why It Matters For SemAxis

Balek et al. address almost the same high-level objective as SemAxis: replace
opaque or high-dimensional text representations with a small collection of
human-readable features that ordinary predictive or rule-learning models can
consume.

The paper is particularly close because its expanded automated workflow asks
an LLM to discover features from dataset metadata, a target definition, and
representative rows, then uses LLM prompts to extract each feature value. In
SemAxis, `SupervisedTransformer` likewise exposes class-contrasting examples
and labels to an LLM, but represents generated features as declarative
hypotheses and measures them using NLI.

## Method

The paper evaluates two feature-generation workflows:

1. User-specified feature list: authors define features such as research rigor,
   grammar, replicability, and novelty; an LLM extracts their values from each
   text.
2. Automated feature discovery: an LLM receives dataset metadata, target
   information, and 40 sampled rows; it generates feature definitions and
   extraction queries, which another LLM stage applies to each instance.

The reported automated discovery stage uses `gpt-4o-2024-08-06`, while feature
value extraction uses `gpt-4o-mini-2024-07-18` through the OpenAI Batch API.
The earlier user-defined scientometric experiment extracts 62 interpretable
features with Llama 2.

The method then trains downstream classifiers and interprets their feature
contributions with SHAP. It also mines action rules on scientometric datasets,
demonstrating why human-readable intermediate columns matter beyond predictive
accuracy.

## Evaluation

- Datasets: CORD-19, M17+, BANKING77, Hate Speech, and Food Hazard.
- Feature comparisons: TF-IDF/BoW, LLM-only features, combined BoW plus LLM
  features; on scientific-text datasets, SciBERT and AutoGluon variants are
  also compared.
- Metrics: accuracy, F1, MAE where appropriate, SHAP analysis, statistical
  association tests, and a small relevance survey.
- Human relevance study: 41 respondents rate 100 automatically discovered
  features; only four receive clearly low mean relevance scores (`1-2`), while
  27 score clearly relevant (`4-5`).

For user-specified scientific features, LLM-only features achieve `0.597`
accuracy on CORD-19 versus `0.625` for SciBERT and `0.625` for TF-IDF; on
M17+, LLM-only features achieve `0.355` versus `0.408` for SciBERT and `0.343`
for TF-IDF. For automatically selected features on Hate Speech, LLM-only
features improve recall relative to TF-IDF in the reported table (`0.52`
versus `0.39`) at lower accuracy and precision.

## Label-Proxy Concern

This paper provides a particularly direct warning for supervised SemAxis. Its
automated feature-discovery prompt includes both the target definition and the
instruction that values matching the target may be extracted as features when
contextually appropriate.

The paper's SHAP discussion for Food Hazard then states that its most important
LLM-generated feature is the model's attempt to assess the target label
directly, illustrated by `hazard_type_biological`. That feature is readable and
predictive, but it is close to a restatement of the evaluated hazard-category
target rather than an explanatory intermediate factor.

This does not establish how much performance is inflated by such features.
It does show that automated interpretable feature generation can intentionally
produce label-proxy features, so their incidence and performance impact must be
reported separately.

## Comparison With Current SemAxis

| Dimension | Balek et al. | SemAxis Today |
|---|---|---|
| Feature discovery | User-specified or LLM-generated from target metadata and sampled rows | LLM-generated from collection or labeled class contrasts |
| Feature representation | Categorical/numerical interpretable columns and extraction prompts | Declarative NLI hypotheses |
| Feature measurement | LLM value extraction | NLI entailment probability |
| Example selection | 40 randomly selected rows for automated discovery | Random, K-Means, or Vote-K under token budget |
| Explanation use | SHAP and action rules | Linear coefficients over NLI features |
| Label-proxy treatment | Target-like features are permitted and observed in SHAP analysis | No audit yet |

## Actionable Findings

1. Strengthen the label-proxy leakage audit candidate. Balek's automated
   workflow makes target-like feature generation explicit, rather than merely
   observing it after generation.
2. When evaluating `sample_method`, include whether representative sampling
   changes the rate of label-proxy hypotheses, not only predictive metrics.
3. Use Balek et al. as a primary related-work citation. Its final journal
   record resolves the report's earlier uncertainty about publication status.
4. Consider action-rule or simple rule-list downstream use as a future
   demonstration of why SemAxis features are useful beyond coefficients.

## Issue Candidate Impact

Balek et al. directly strengthens
[`../ideas/label-proxy-leakage-audit.md`](../ideas/label-proxy-leakage-audit.md).
It does not require a new candidate in this PR: sampling-dependent proxy rates
can be included when defining the audit experiment, and rule-learning use can
remain a later product/research direction.

## Sources Checked

- OpenAlex journal and preprint records queried with `pyalex`, accessed
  2026-05-26:
  <https://openalex.org/W4414681764>
- Public arXiv full text, accessed 2026-05-26:
  <https://arxiv.org/abs/2409.07132>
