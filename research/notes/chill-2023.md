# CHiLL (McInerney et al., 2023)

## Citation

Denis McInerney, Geoffrey Young, Jan-Willem van de Meent, and Byron Wallace.
2023. *CHiLL: Zero-shot Custom Interpretable Feature Extraction from Clinical
Notes with Large Language Models*. Findings of the Association for
Computational Linguistics: EMNLP 2023, pages 8477-8494. DOI:
[`10.18653/v1/2023.findings-emnlp.568`](https://doi.org/10.18653/v1/2023.findings-emnlp.568).

- ACL Anthology: `2023.findings-emnlp.568`
- OpenAlex: `W4389520444`
- Local PDF cache: `../pdfs/chill-2023.pdf`
- Official page: <https://aclanthology.org/2023.findings-emnlp.568/>

## Why It Matters For SemAxis

CHiLL demonstrates that natural-language high-level features can be used as
the columns of a small linear predictor while remaining inspectable to domain
experts. It is a direct precedent for SemAxis's claim that readable features
and linear coefficients are valuable.

CHiLL differs from SemAxis at the feature-definition boundary. A domain expert
writes the feature questions, and an LLM performs zero-shot extraction of
their values. SemAxis generates the hypothesis bank automatically before NLI
scoring it. This means SemAxis removes expert feature-writing effort but must
audit automatically generated features for redundancy, circular label proxies,
and stability.

## Method

CHiLL asks an expert to specify natural-language queries for clinically
meaningful intermediate features. An instruction-tuned language model
(`Flan-T5-XXL` in the reported experiments) estimates whether each query is
true for a clinical note.

The paper evaluates both:

- Binary values derived from the model's `yes`/`no` decision.
- Continuous values derived from normalized probability assigned to `yes`
  versus `no`.

A small logistic model then predicts the downstream clinical target from the
inferred intermediate feature vector. Continuous inferred features outperform
binary versions in the reported downstream comparisons, making score-valued
human-readable features particularly relevant to SemAxis's NLI probabilities.

## Evaluation

- Data: MIMIC-III clinical notes and MIMIC-CXR radiology reports.
- Tasks: phenotyping, 30-day readmission, in-hospital mortality, and
  multi-label chest X-ray report classification.
- Feature references: ICD codes for MIMIC-III; radiologist-specified custom
  feature queries and manually annotated examples for the X-ray setting.
- Baselines: ClinicalBERT/BERT-style representations, TF-IDF, reference
  high-level features, and direct zero-shot downstream prediction.
- Metrics: AUROC for downstream classification and continuous feature
  extraction, plus F1/precision/recall for binary feature extraction.

The paper reports that continuous inferred high-level features contain
significant signal and perform comparably to or better than reference
high-level features on most reported tasks except phenotyping. They remain
below BERT and full TF-IDF in downstream performance, while using only
approximately `10-105` readable features instead of tens of thousands of
lexical or dense predictors. On chest X-ray classification, learned feature
rankings have AUC greater than `0.5` against radiologist relevance judgments.

## Interpretability Warning

CHiLL explicitly distinguishes a readable predictor from a useful
explanation. In its limitations, the paper considers a mortality-prediction
feature question equivalent to asking whether the patient is at risk of death.
Such a feature may correlate strongly with the target, but it essentially
paraphrases the downstream task and does little to explain it.

This is the same failure mode captured in this repository as
**label-proxy leakage**: a high-level hypothesis can be legible and predictive
while remaining circular as an explanatory feature.

CHiLL also notes that an inferred feature value may itself be incorrect.
Inspecting linear coefficients is only meaningful if the measurement step
faithfully evaluates each intended feature. For SemAxis, NLI replaces an LLM
feature extractor, but it does not eliminate the need to test whether
entailment scores reflect the stated hypotheses reliably.

## Comparison With Current SemAxis

| Dimension | CHiLL | SemAxis Today |
|---|---|---|
| Feature definition | Expert-written natural-language queries | LLM-generated declarative hypotheses |
| Feature measurement | Flan-T5 `yes`/`no` probabilities | NLI entailment probability |
| Supervision for feature bank | Expert knowledge | Text collection and optional labels |
| Downstream model | Small linear classifier | Any sklearn downstream estimator |
| Explanatory safeguard | Expert controls feature wording; limitations still identify circular queries | No audit for generated label proxies yet |
| Feature correctness check | ICD-code and radiologist reference comparisons | Not yet implemented |

## Actionable Findings

1. Strengthen the label-proxy leakage audit candidate: CHiLL independently
   describes a downstream-task paraphrase as predictive but unhelpful for
   interpretability.
2. Consider a future user-specified hypothesis mode for SemAxis. It would make
   CHiLL-like comparisons possible and allow domain experts to supply features
   while retaining NLI-based measurement.
3. Include a measurement-faithfulness evaluation in longer-term benchmark
   planning. Natural-language feature labels alone are not evidence that the
   scores are correct for each text.

## Issue Candidate Impact

CHiLL strengthens
[`../ideas/label-proxy-leakage-audit.md`](../ideas/label-proxy-leakage-audit.md)
without expanding its scope. A separate future candidate could cover
user-specified hypotheses and feature-measurement validation after the initial
audit work is defined.

## Sources Checked

- ACL Anthology official page and PDF, accessed 2026-05-26:
  <https://aclanthology.org/2023.findings-emnlp.568/>
- OpenAlex record queried with `pyalex`, accessed 2026-05-26:
  <https://openalex.org/W4389520444>
