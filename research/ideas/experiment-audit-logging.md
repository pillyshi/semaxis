# Issue Candidate: Add Experiment Audit Logging For Generated Hypothesis Runs

## Status

Promoted to GitHub issue:
[#54: research: add experiment audit logging for generated hypothesis runs](https://github.com/pillyshi/semaxis/issues/54).

## Motivation

SemAxis generates natural-language hypotheses, scores each text-hypothesis
pair with an NLI model, and trains downstream estimators over the resulting
feature matrix. The exact hypotheses, prompts, labels, sampled examples, NLI
backbone, and fold boundaries are therefore part of the experimental result,
not incidental runtime details.

Current issue candidates already need this information:

- Label-proxy leakage auditing requires exact hypothesis wording, direct NLI
  baseline wording, flagged hypotheses, and before/after filtering metrics.
- Semantic deduplication needs generated count, retained count, pruned
  hypotheses, similarity decisions, and NLI-scoring work avoided.
- Sampling benchmarks need selected example identifiers, `sample_method`, seed,
  and class-contrast context.

Without a structured run artifact, results are hard to reproduce or audit.
Console logging is not enough: it is not stable, queryable, or suitable for
fold-level comparisons.

## Evidence

- Ma et al. 2021. *Issues with Entailment-based Zero-shot Text Classification*.
  The paper shows that entailment-based zero-shot classification varies across
  datasets and NLI fine-tuning sources, and that raw sentence-pair scoring can
  be competitive. This makes NLI backbone and exact hypothesis wording part of
  the result.
- [`../notes/ma-2021.md`](../notes/ma-2021.md) records the NLI instability and
  lexical-pattern concerns.
- Yin et al. 2019. *Benchmarking Zero-shot Text Classification*. The paper
  shows sensitivity to label-word and definition-style hypotheses, motivating
  retention of exact wording for direct NLI baselines.
- [`../notes/yin-2019.md`](../notes/yin-2019.md) records the entailment
  formulation and hypothesis-wording sensitivity.
- Cosma et al. 2026. *Automatic Prompt Optimization for Dataset-Level Feature
  Discovery*. The paper evaluates generated feature sets using dataset-level
  performance and interpretability feedback, including explicit label-leakage
  penalties.
- [`../notes/cosma-2026.md`](../notes/cosma-2026.md) records the feature-set
  feedback loop and label-proxy warning.
- Gera et al. 2022. *Zero-Shot Text Classification with Self-Training*. The
  paper adapts NLI zero-shot classifiers using unlabeled target-task examples,
  making adaptation state, unlabeled-pool identity, template wording, and
  pseudo-label selection part of the experimental result.
- [`../notes/gera-2022.md`](../notes/gera-2022.md) records the self-training
  setup and its protocol implications for direct NLI baselines.
- Koh et al. 2020. *Concept Bottleneck Models*. The paper shows that
  bottleneck-style interpretability depends on concept measurement quality,
  downstream reliance on the bottleneck, and intervention behavior, not only
  readable concept names.
- [`../notes/koh-2020.md`](../notes/koh-2020.md) records the CBM framing and
  its relevance to SemAxis's generated feature matrix.
- Liu et al. 2022. *What Makes Good In-Context Examples for GPT-3?* The paper
  shows that selected in-context examples and retrieval encoder choice can
  substantially change LLM behavior, motivating explicit logging of selected
  row ids, embedding model, and sample order.
- [`../notes/liu-2022.md`](../notes/liu-2022.md) records the KATE
  retrieval setup and its relevance to SemAxis sampling experiments.
- Su et al. 2022. *Selective Annotation Makes Language Models Better Few-Shot
  Learners*. The paper introduces Vote-K and shows that selected example pools,
  graph parameters, retrieval strategy, and prompt order affect ICL performance
  and stability.
- [`../notes/su-2022.md`](../notes/su-2022.md) records the Vote-K algorithm
  and its limits when transferred to SemAxis fit-time sampling.
- Zhang et al. 2023. *IDEAL*. The paper shows that influence-driven graph
  selection can outperform Vote-K and that selection diagnostics such as graph
  coverage, outliers, and prompt-order stability matter.
- [`../notes/zhang-2023.md`](../notes/zhang-2023.md) records the IDEAL method
  and motivates a SemAxis sampling benchmark.
- Cawley and Talbot 2010. *On Over-fitting in Model Selection and Subsequent
  Selection Bias in Performance Evaluation*. Their selection-bias argument
  supports recording fold-local generation, selection, and tuning decisions
  rather than treating generated features as fixed global preprocessing.
- Existing issue candidates
  [`label-proxy-leakage-audit.md`](label-proxy-leakage-audit.md) and
  [`semantic-hypothesis-dedup.md`](semantic-hypothesis-dedup.md) both require
  per-run generated-feature metadata for their acceptance criteria.

## Proposed Scope

- Define a versioned artifact schema for SemAxis experiment runs.
- Store one machine-readable artifact per fit, fold, or evaluation run using a
  durable format such as JSON or JSONL.
- Capture hypothesis-generation metadata:
  - exact prompt or prompt template identifier;
  - class labels or anonymized labels shown to the generator;
  - generator model, temperature, seed where available, and provider/version;
  - selected example identifiers or stable row indices;
  - `sample_method`, embedding model, similarity metric, sample budget, prompt
    order, graph or clustering parameters, coverage/outlier diagnostics, and
    fold identifier.
- Capture feature metadata:
  - generated hypotheses in exact wording;
  - supervised contrast metadata from `feature_meta_` where available;
  - retained, pruned, or flagged hypotheses and reasons;
  - downstream feature columns actually exposed to the estimator, to make
    bottleneck-side-channel assumptions auditable;
  - direct NLI baseline hypotheses when used.
- Capture measurement metadata:
  - NLI backbone name and revision;
  - score definition, such as entailment probability or entailment-margin;
  - adaptation method, unlabeled pool, and pseudo-label selection rule when an
    NLI baseline or scorer is fine-tuned;
  - number of text-hypothesis pairs scored;
  - wall-clock time and cache hit/miss counts if available.
- Capture evaluation metadata:
  - downstream estimator and metric names;
  - before/after filtering or deduplication metrics;
  - fold/seed identifiers and dataset split references.

## Acceptance Criteria

- A documented schema exists for generated-hypothesis run artifacts.
- Label-proxy leakage audit runs can be reconstructed from saved artifacts:
  generated hypotheses, direct baseline wording, flags, reasons, and metrics
  are all present.
- Deduplication experiments can report generated count, retained count, pruned
  hypotheses, and NLI pair count before and after pruning from artifacts.
- The logging path is opt-in or explicitly scoped so ordinary transformer use
  does not write unexpected files by default.
- Tests cover schema serialization for supervised and unsupervised runs with
  deterministic toy metadata.
- Sensitive raw texts are not stored by default; artifacts store row ids,
  hashes, snippets only when explicitly enabled, or user-provided identifiers.

## Out Of Scope

- Integrating a full experiment tracker such as MLflow, Weights & Biases, or
  DVC.
- Persisting every NLI score by default.
- Changing model behavior based on logged artifacts.
- Replacing Python `logging` for normal runtime diagnostics.
- Designing a complete benchmark suite.

## Open Design Questions

- Should artifacts live under `research/runs/`, a user-specified output
  directory, or an sklearn-style callback/export hook?
- Should the transformer expose an `audit_log_` attribute, write directly to
  disk, or return artifacts through a separate evaluator utility?
- How should row identity be represented when input data has no stable id?
- Should prompt text be stored verbatim, hashed, or both?
- Which metadata belongs in a stable public schema versus an experimental
  extension field?
