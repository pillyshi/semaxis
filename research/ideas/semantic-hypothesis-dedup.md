# Issue Candidate: Add Semantic Deduplication For Generated Hypotheses

## Status

Promoted to GitHub issue:
[#19: prune redundant hypotheses before NLI scoring](https://github.com/pillyshi/semaxis/issues/19).

## Motivation

`SupervisedTransformer` and `UnsupervisedTransformer` currently store all
hypotheses returned by the generator and score every retained hypothesis with
NLI. Semantically repeated hypotheses therefore create redundant output
columns, make coefficient-based explanations noisier, and increase NLI
inference cost.

Downstream feature selection can reduce redundant columns before the final
predictor is fitted, but it operates after the feature matrix has been
computed. It therefore does not avoid the NLI scoring cost of redundant
hypotheses. This candidate focuses on pruning between generation and NLI
measurement; it complements rather than replaces downstream feature
selection.

FELIX addresses a closely related problem by embedding generated feature
schemas, clustering semantically related candidates, and retaining a
representative feature from each cluster.

## Evidence

- Malberg, Mosca, Groh. 2024. *FELIX: Automatic and Interpretable Feature
  Engineering Using LLMs*. Sec. 3.2 describes feature consolidation by
  embedding, HDBSCAN clustering, and representative selection.
- [`../notes/felix-2024.md`](../notes/felix-2024.md) records the direct
  comparison with SemAxis.
- Ludan et al. 2023. *Interpretable-by-Design Text Understanding with
  Iteratively Generated Concept Bottleneck*. Its human evaluation reports
  redundancy as the most frequent generated-concept issue (`25%` on average)
  and calls for filtering redundant and leaky concepts.
- [`../notes/tbm-2023.md`](../notes/tbm-2023.md) records TBM's generation and
  evaluation findings relevant to this candidate.
- Tavares. 2025. *LogiPart*. Its hierarchy builder blocklists hypotheses
  already used or rejected at higher tree levels to avoid repeated semantic
  splits, providing a nearby NLI-based precedent for redundancy control.
- [`../notes/logipart-2025.md`](../notes/logipart-2025.md) records the
  hypothesis-generation and NLI-assignment comparison with SemAxis.
- Current implementation appends generated hypotheses without a consolidation
  step in `src/semaxis/supervised.py` and `src/semaxis/unsupervised.py`.

## Proposed Scope

- Add an optional post-generation pruning strategy for `features_`.
- Apply pruning before NLI measurement so redundant hypotheses do not incur
  avoidable scoring work.
- Begin with deterministic embedding-similarity deduplication using a
  configurable similarity threshold; avoid requiring HDBSCAN in the first
  implementation.
- Ensure `SupervisedTransformer.feature_meta_` stays aligned with retained
  hypotheses.
- Expose enough retained/pruned information to report redundancy and NLI cost.
- Add focused unit tests for deterministic pruning, disabled behavior, and
  `feature_meta_` alignment.

## Acceptance Criteria

- Users can enable semantic hypothesis deduplication through transformer
  configuration without changing existing default behavior.
- Enabled deduplication reduces the number of hypotheses submitted to NLI
  scoring when near-duplicates are generated.
- Duplicate or near-duplicate hypotheses are removed deterministically under a
  mocked embedding matrix.
- Supervised feature metadata remains parallel to the retained feature list.
- Tests cover both supervised and unsupervised transformers.
- A small comparison records generated count, retained count, and number of
  NLI-scored columns with and without deduplication.

## Out Of Scope

- Making deduplication the default.
- Reproducing FELIX end-to-end.
- Implementing HDBSCAN or full feature clustering before a simpler threshold
  method is evaluated.
- Running a multi-dataset benchmark.
- Replacing downstream predictive feature selection.

## Open Design Choice

Deduplication can operate globally or within each supervised class contrast.
Global pruning reduces cost most aggressively; per-contrast pruning preserves
the provenance encoded by `feature_meta_`. This should be decided before
implementation.
