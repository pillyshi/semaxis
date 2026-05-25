# FELIX (Malberg, Mosca, Groh, 2024)

## Citation

Simon Malberg, Edoardo Mosca, and Georg Groh. 2024. *FELIX: Automatic and
Interpretable Feature Engineering Using LLMs*. ECML PKDD 2024, Lecture Notes
in Computer Science, pages 230-246. DOI:
[`10.1007/978-3-031-70359-1_14`](https://doi.org/10.1007/978-3-031-70359-1_14).

- OpenAlex: `W4402011475`
- Local PDF cache: `../pdfs/felix-2024.pdf`
- Public PDF: <https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2024/lncs14944/lncs14944231.pdf>

## Why It Matters For SemAxis

FELIX is a direct competitor for the broad claim that an LLM can create
human-readable text features for an ordinary downstream classifier. It
generates features from labeled text pairs, consolidates semantically similar
features, and assigns a structured feature value to each example.

SemAxis should therefore not claim novelty for automatic interpretable text
feature generation alone. Its defensible distinction is that it generates
declarative hypotheses and delegates value assignment to an NLI scorer in a
sklearn-compatible transformer that can be fitted within a cross-validation
fold.

## Method

FELIX has three phases:

1. Feature generation: an LLM sees labeled pairs from different classes and
   proposes discriminative feature schemas.
2. Feature selection: feature schemas are embedded, clustered with HDBSCAN,
   and reduced by keeping the feature nearest each cluster centroid.
3. Value assignment: an LLM assigns a numerical or categorical value for each
   selected feature and input example.

The feature-selection stage is particularly relevant: it explicitly treats
semantic redundancy as both a quality problem and a cost problem.

## Evaluation

- Tasks: sentiment, hate speech, Amazon reviews, fake news, and
  human-vs-machine-written scientific papers.
- Baselines: TF-IDF, text embeddings, zero-shot GPT-3.5/GPT-4, and fine-tuned
  RoBERTa.
- Models over extracted features: logistic regression and random forest.
- Main metric: F1.
- Sample-efficiency evaluation varies training examples across
  `10`, `20`, `50`, and `100`.
- Generalization evaluation trains and tests across Amazon and Yelp domains.

The reported average F1 for GPT-4 FELIX variants is approximately `90-91%`
over five datasets, above the reported TF-IDF, embedding, zero-shot GPT-4,
and RoBERTa averages in that setup. This should be treated as motivation for a
comparison, not as evidence that SemAxis will behave similarly.

## Comparison With Current SemAxis

| Dimension | FELIX | SemAxis Today |
|---|---|---|
| Generated object | Numerical/categorical feature schemas | Declarative NLI hypotheses |
| Generation supervision | Labeled example pairs | Unsupervised collection or labeled class contrast |
| Feature consolidation | Embedding + HDBSCAN + centroid representative | No post-generation semantic deduplication |
| Value assignment | LLM prompt per example/feature set | NLI cross-encoder entailment score |
| Downstream use | LR / RF | Any sklearn downstream estimator |
| CV safety | Train/test setup described in experiments | Transformer generation occurs in `fit`, enabling fold-local use |

Relevant SemAxis code:

- `src/semaxis/supervised.py`: generates and directly extends `features_`.
- `src/semaxis/unsupervised.py`: generates and directly assigns `features_`.
- `src/semaxis/sampling.py`: embedding-aware selection exists for input texts,
  not generated hypotheses.

## Actionable Findings

1. Add post-generation semantic deduplication or pruning of hypotheses.
   Redundant hypotheses currently become redundant NLI columns and repeat NLI
   evaluation work.
2. Include FELIX in related-work positioning and use its baselines when
   defining a SemAxis benchmark.
3. Measure feature redundancy and NLI cost alongside classification quality;
   feature count alone does not show whether the generated bank is useful.

## First Issue Candidate

Drafted as [`../ideas/semantic-hypothesis-dedup.md`](../ideas/semantic-hypothesis-dedup.md):
add an optional, deterministic hypothesis deduplication stage and report its
effect on retained features and NLI computation.

## Sources Checked

- FELIX published preprint PDF, sections 3 and 4, accessed 2026-05-26:
  <https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2024/lncs14944/lncs14944231.pdf>
- TUM publication record, accessed 2026-05-26:
  <https://portal.fis.tum.de/de/publications/felix-automatic-andinterpretable-feature-engineering-using-llms/>
- OpenAlex record queried with `pyalex`, accessed 2026-05-26:
  <https://openalex.org/W4402011475>
