# Issue Candidate: Benchmark Sampling Strategies For Hypothesis Generation

## Status

Promoted to GitHub issue:
[#53: research: benchmark sampling strategies for hypothesis generation](https://github.com/pillyshi/semaxis/issues/53).

## Motivation

SemAxis uses a limited sample of training texts to prompt LLM hypothesis
generation. The selected examples shape which concepts or hypotheses the LLM
sees, so `sample_method` is an experimental variable rather than a convenience
setting.

Current SemAxis sampling includes `random`, `kmeans`, and `votek`. Literature
supports embedding-aware selection, but also shows that no single selection
strategy should be assumed best across tasks. Vote-K is a strong baseline, yet
its original setting differs from SemAxis and later work identifies practical
limitations.

This candidate defines an evaluation, not an immediate default change.

## Evidence

- Liu et al. 2022. *What Makes Good In-Context Examples for GPT-3?* Retrieved
  examples and embedding model choice substantially affect LLM behavior.
- [`../notes/liu-2022.md`](../notes/liu-2022.md) records the KATE retrieval
  setup and transfer caveats for SemAxis.
- Su et al. 2022. *Selective Annotation Makes Language Models Better Few-Shot
  Learners*. Vote-K selects diverse and representative examples and improves
  few-shot ICL under annotation budgets.
- [`../notes/su-2022.md`](../notes/su-2022.md) records Vote-K and its
  dependence on the original selective-annotation plus retrieval setting.
- Zhang et al. 2023. *IDEAL*. The paper critiques Vote-K as non-end-to-end,
  potentially outlier-prone, and lacking theory, then proposes influence-driven
  graph selection.
- [`../notes/zhang-2023.md`](../notes/zhang-2023.md) records IDEAL and its
  relevance to SemAxis sampling evaluation.
- LogiPart evaluates random, K-Means, and Vote-K variants in a nearby
  LLM-generated predicate plus NLI-assignment architecture.
- [`../notes/logipart-2025.md`](../notes/logipart-2025.md) records the
  sampling and cost comparison with SemAxis.

## Proposed Scope

- Define a benchmark comparing SemAxis sampling strategies under fixed prompt
  token budgets.
- Include current methods: `random`, `kmeans`, and `votek`.
- Include at least one coverage-oriented or influence-inspired baseline, such
  as facility-location greedy selection, before implementing more complex IDEAL
  diffusion.
- Evaluate both supervised and unsupervised hypothesis generation when
  feasible.
- Record selected example ids, prompt order, embedding model, similarity
  metric, graph/clustering parameters, and token budget.
- Report generated-hypothesis quality and cost, not only downstream metrics.

## Suggested Measurements

- Downstream predictive metric on at least one supervised task.
- Total generated hypotheses and retained hypotheses.
- Semantic redundancy among generated hypotheses.
- Label-proxy rate for supervised generation.
- Stability of generated hypotheses across seeds or folds.
- NLI pair count and wall-clock scoring cost.
- Pairwise similarity, coverage, and outlier diagnostics for selected examples.
- Prompt token count and truncation behavior.

## Acceptance Criteria

- The benchmark compares at least three existing sampling strategies under the
  same prompt budget.
- Selected-example metadata is saved through the experiment audit logging
  candidate or an equivalent artifact.
- Results report at least one feature-quality metric in addition to downstream
  predictive performance.
- Vote-K is evaluated as a baseline, not promoted to a default solely from ICL
  literature.
- Any proposed new default or new sampling method is justified by SemAxis
  results, not only by external ICL results.

## Out Of Scope

- Reimplementing full IDEAL as the first step.
- Combining sampling evaluation with semantic deduplication implementation.
- Claiming a globally best sampling method across all tasks.
- Running a large multi-dataset benchmark before a small controlled benchmark
  demonstrates useful signal.

## Open Design Questions

- Should supervised sampling happen per class, per class contrast, or globally?
- Should the benchmark compare selected text diversity or generated hypothesis
  diversity as the primary sampling diagnostic?
- Should token budget or example count be the controlling variable?
- Which embedding model should be treated as the default baseline?
- How should long documents be represented when only prefixes fit in prompts?
