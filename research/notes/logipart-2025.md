# LogiPart (Tavares, 2025)

## Citation

Tiago Fernandes Tavares. 2025. *LogiPart: Local Large Language Models for
Data Exploration at Scale with Logical Partitioning*. arXiv:2509.22211,
current v3 dated 2026-02-17. DOI:
[`10.48550/arXiv.2509.22211`](https://doi.org/10.48550/arXiv.2509.22211).

- OpenAlex: `W7083679346`
- Local PDF cache: `../pdfs/2509.22211v3.pdf`
- arXiv record: <https://arxiv.org/abs/2509.22211>

The arXiv record indicates that v3 is a substantial rewrite of an earlier
title. This note uses the v3 title and architecture reflected in the cached
PDF.

## Why It Matters For SemAxis

LogiPart is the closest reviewed paper to SemAxis at the mechanics level:
an LLM generates concise, evaluable natural-language assertions and an NLI
model evaluates texts against those assertions. It also evaluates K-Means and
Vote-K sampling strategies, which now exist in SemAxis.

The product differs substantially. LogiPart creates a hierarchical binary
partition for exploratory data organization; each hypothesis becomes a tree
split. SemAxis creates a flat `n_texts x n_features` matrix for arbitrary
sklearn prediction. LogiPart therefore supports the NLI featureization premise
but is not an equivalent classifier baseline.

## Method

At each tree node, LogiPart:

1. Embeds documents and samples texts for hypothesis generation.
2. Prompts an LLM to produce a single taxonomic assertion intended to be true
   for one subset and false for another.
3. Scores sampled documents against that assertion using zero-shot NLI.
4. Uses graph-based label propagation to assign remaining documents efficiently.
5. Accepts a non-trivial split and recursively repeats the procedure.

The prompt constrains generated hypotheses to conceptual differences in human
intent or content rather than surface words or punctuation. It also provides a
blocklist of previously used or rejected hypotheses to reduce redundant
splits. The v3 implementation reports using
`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` as its NLI backbone and labels
entailment based on an entailment-versus-contradiction score.

## Sampling And Cost

LogiPart tests random, K-Means, Vote-K, and bisection-oriented sampling
configurations for the LLM and NLI stages. Its preferred AG News configuration
uses Vote-K for hypothesis-generation samples and K-Means for NLI assignment
samples with `gpt-oss:20b`.

The framework confines LLM generation to a fixed-size sample at each node, so
generative input-token cost is approximately constant with respect to corpus
size. NLI evaluation still grows with the sampled assignment workload, while
label propagation is used to avoid applying NLI to every document.

This is relevant to SemAxis because its NLI work scales with the cross product
of texts and retained features. LogiPart reinforces the need to report
generation cost and NLI assignment cost separately rather than only counting
LLM calls.

## Evaluation

- Corpora: AG News, 20 Newsgroups, Wikipedia, and US Bills, totaling roughly
  140,000 documents.
- Structure: trees up to height `6`, with splitting stopped for nodes below
  `200` documents.
- Metrics: NMI, ARI, accuracy, and F1 where thematic labels are informative;
  inverse-logic routing validation and qualitative predicate auditing where
  discovered axes may be orthogonal to thematic labels.
- Reported result: individual taxonomic bisections reach average per-node
  routing accuracy of up to `96%` in its inverse-logic validation setting.

The evaluation purpose differs from SemAxis classification: LogiPart argues
that a useful discovered partition may not reproduce a pre-existing topic
label. That is relevant for SemAxis's unsupervised mode, where quality cannot
be assessed solely through downstream label prediction.

## Limitations Relevant To SemAxis

LogiPart identifies the NLI model as a central dependency: efficient assignment
can still be miscalibrated for abstract or non-discriminative hypotheses. It
also assumes that fixed-length samples contain enough document semantics, which
may fail for long documents where relevant meaning is not front-loaded.

Both limitations transfer directly:

- SemAxis cannot treat NLI score columns as faithful merely because hypotheses
  are readable.
- The quality of generated hypotheses depends on what input sampling exposes to
  the LLM, particularly for long or heterogeneous corpora.

## Comparison With Current SemAxis

| Dimension | LogiPart | SemAxis Today |
|---|---|---|
| Generated object | One natural-language predicate per tree node | Multiple hypotheses per transformer fit |
| Measurement | NLI entailment/contradiction assignment | NLI entailment probability |
| Sampling | Random, K-Means, Vote-K, bisection variants | Random, K-Means, Vote-K |
| Output | Hierarchical logical partition | Flat sklearn feature matrix |
| Redundancy handling | Prompt blocklist for used/rejected predicates | Prompt instruction only; no retained-bank filtering |
| Scaling tactic | Sampled NLI plus label propagation | Score all text-feature pairs |

## Actionable Findings

1. Strengthen semantic hypothesis deduplication: LogiPart independently avoids
   repeated hypotheses in a nearby NLI-based architecture.
2. When defining a sampling benchmark, evaluate `random`, `kmeans`, and
   `votek` not only on prediction but on feature stability, redundancy, and
   NLI scoring cost.
3. Retain measurement-faithfulness evaluation as a future requirement:
   LogiPart explicitly notes NLI miscalibration risk for abstract hypotheses.
4. Position LogiPart in related work as a hierarchical exploratory partitioning
   approach rather than claiming it provides the same flat featureization API.

## Issue Candidate Impact

LogiPart strengthens
[`../ideas/semantic-hypothesis-dedup.md`](../ideas/semantic-hypothesis-dedup.md).
It also motivates a later benchmark/evaluation candidate covering sampling
strategy, cost, and NLI measurement stability, but that scope should not be
folded into the first deduplication implementation issue.

## Sources Checked

- Cached arXiv v3 PDF, dated 2026-02-17 and accessed 2026-05-26:
  <https://arxiv.org/abs/2509.22211>
- OpenAlex record queried with `pyalex`, accessed 2026-05-26:
  <https://openalex.org/W7083679346>
