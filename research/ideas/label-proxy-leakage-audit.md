# Issue Candidate: Audit Label-Proxy Leakage In Supervised Hypotheses

## Status

Promoted to GitHub issue:
[#20: audit label-proxy leakage in supervised hypotheses](https://github.com/pillyshi/semaxis/issues/20).

## Terminology

This candidate follows TBM's use of **leakage**: a generated concept or
hypothesis directly performs the labeling task instead of representing a more
specific explanatory factor. To avoid confusion with train/test data leakage,
this document refers to it as **label-proxy leakage**.

Example for sentiment classification:

- Explanatory hypothesis: `This text complains about poor build quality.`
- Label-proxy hypothesis: `This text expresses negative sentiment.`

Example for hate-speech classification:

- Explanatory hypothesis: `This text expresses hostility toward a protected group.`
- Label-proxy hypothesis: `This text contains hate speech.`

Fold-local feature generation can prevent test-set contamination while still
allowing label-proxy leakage.

## Motivation

`SupervisedTransformer` sends labeled positive and negative text groups,
including their label names, to the generator. It then accepts all returned
hypotheses into `features_`. When labels are semantically informative, such as
`positive`, `negative`, or `hate_speech`, a generator can emit hypotheses that
merely restate the target class.

Such features may be predictive, but they weaken SemAxis's claim that the
resulting feature matrix supplies useful intermediate explanations. They may
also make the method behave like direct zero-shot NLI classification under a
more expensive feature-generation process.

## Evidence

- Ludan et al. 2023. *Interpretable-by-Design Text Understanding with
  Iteratively Generated Concept Bottleneck*. The paper defines leakage as a
  concept that leaks the labeling task and reports it for `15%` of generated
  concepts on average in its human concept audit. This is a concept incidence
  rate, not a measured change in predictive performance.
- TBM's repeated CEBaB runs observe that two of its best-performing final
  models include label-leaking concepts, motivating an explicit performance
  ablation rather than assuming leakage is harmless.
- [`../notes/tbm-2023.md`](../notes/tbm-2023.md) summarizes these findings.
- McInerney et al. 2023. *CHiLL*. Its limitations discuss a mortality query
  equivalent to asking whether the patient is at risk of death: it can be
  predictive while paraphrasing the downstream task and adding little
  interpretability.
- [`../notes/chill-2023.md`](../notes/chill-2023.md) records CHiLL's
  expert-specified feature setting and interpretability warning.
- Balek et al. 2025. *LLM-based feature generation from text for interpretable
  machine learning*. Its automated feature-discovery prompt permits features
  derived from target-matching values; in the Food Hazard SHAP example the top
  feature is `hazard_type_biological`, directly approximating the evaluated
  hazard-category target.
- [`../notes/balek-2025.md`](../notes/balek-2025.md) records this automated
  discovery workflow and its similarity to supervised SemAxis generation.
- Yin et al. 2019. *Benchmarking Zero-shot Text Classification*. The paper
  turns class labels into NLI hypotheses and shows that label wording
  (`word`, `definition`, or a combination) changes results across tasks and
  entailment models. This makes direct zero-shot NLI the nearest baseline when
  generated features merely paraphrase labels.
- [`../notes/yin-2019.md`](../notes/yin-2019.md) records the entailment
  formulation and hypothesis-wording sensitivity.
- Current `src/semaxis/supervised.py` passes class labels to generation and
  stores all returned hypotheses without a leakage audit.

## Proposed Scope

- Define an auditable label-proxy leakage taxonomy for generated supervised
  hypotheses.
- Add an experiment or evaluation utility that reports potentially leaky
  hypotheses and a per-run proxy rate.
- Compare semantic class labels against anonymized generation labels, such as
  `class A` and `class B`, while preserving the true labels for the downstream
  classifier.
- Compare downstream performance before and after filtering flagged
  label-proxy hypotheses.
- Include a direct zero-shot NLI label-hypothesis baseline where applicable.

## Suggested Measurements

- Total generated hypotheses.
- Number and fraction flagged as label-proxy leakage.
- Predictive metric before and after filtering flagged hypotheses.
- Predictive metric when feature generation receives anonymized labels.
- Overlap between generated hypotheses and direct class-label hypotheses.
- Variation of the above across folds or seeds.
- Sensitivity of proxy rate to `sample_method` when more than one selection
  strategy is included in the audit.

## Acceptance Criteria

- The audit explicitly distinguishes label-proxy leakage from test-fold
  contamination.
- At least one supervised text classification setup is evaluated with
  semantic labels and anonymized generation labels.
- The result reports flagged-hypothesis rate and before/after filtering
  performance rather than inferring an effect from incidence alone.
- Direct zero-shot NLI is included or its exclusion is justified.
- The direct baseline and generated-hypothesis comparison retains the actual
  hypothesis wording used for measurement.
- Findings are recorded in a research note before implementing automatic
  filtering as a default behavior.

## Out Of Scope

- Automatically blocking all high-level sentiment or toxicity hypotheses.
- Treating hypothesis filtering as a substitute for fold-local fitting.
- Combining this audit with semantic deduplication implementation.
- Claiming leakage inflates accuracy unless an ablation demonstrates it.

## Design Questions

- Should proxy detection begin as human review, LLM-assisted review, lexical
  label-overlap rules, or a combination?
- How should legitimate high-level features be distinguished from circular
  restatements of a task label?
- Should semantic labels be hidden only from generation, or should the prompt
  compare anonymized classes while retaining task metadata elsewhere?
