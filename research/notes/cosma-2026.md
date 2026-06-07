# Cosma et al. (2026)

## Citation

Adrian Cosma, Oleg Szehr, David Kletz, Alessandro Antonucci, and Olivier
Pelletier. 2026. *Automatic Prompt Optimization for Dataset-Level Feature
Discovery*. arXiv:2601.13922. DOI:
[`10.48550/arXiv.2601.13922`](https://doi.org/10.48550/arXiv.2601.13922).

- OpenAlex: `W7125115128`
- Local PDF cache: `../pdfs/2601.13922v1.pdf`
- arXiv record: <https://arxiv.org/abs/2601.13922>

The arXiv record currently lists v1, submitted 2026-01-20. No peer-reviewed
venue was identified during this reading.

## Why It Matters For SemAxis

Cosma et al. are close to SemAxis at the system boundary: given a labeled text
corpus, they induce a global set of interpretable and discriminative feature
definitions, extract feature values for texts, and train a simple downstream
classifier over those extracted features.

The main distinction is the measurement step and optimization loop. Their
pipeline uses language-model agents both to propose feature definitions and to
extract structured feature values, then optimizes the feature-proposal prompt
using dataset-level feedback. SemAxis generates declarative hypotheses and
uses NLI entailment scores as the feature values, keeping the measurement step
more constrained and sklearn-transformer shaped.

This paper should be treated as direct related work rather than a peripheral
prompt-optimization citation. It also strengthens SemAxis's need to report
label-proxy leakage separately from predictive performance.

## Method

The framework optimizes a prompt for a `FEATUREPROPOSER` module. A candidate
prompt combines instructions with a sampled set of labeled texts and induces a
global feature schema for the whole corpus, not a per-example prediction.

The system then evaluates the proposed schema through several modules:

1. `FEATUREPROPOSER` outputs feature names, types, descriptions, and extraction
   prompts.
2. `EXTRACTOR` maps each text to realized feature values under that schema.
3. A linear classifier is trained on the extracted feature matrix.
4. `PERFORMANCEFEEDBACK` summarizes dataset-level metrics such as F1, SHAP,
   mutual information, and coverage.
5. `INTERPRETABILITYSCORER` scores whether the feature set is readable,
   human-worded, understandable, meaningful, trackable, and non-leaky.
6. `REFLECTIVEPROPOSER` uses the feedback to propose refined instructions.

The final search combines instruction proposals and sampled example sets using
Bayesian optimization. This differs from ordinary prompt optimizers such as
MIPRO because feedback is computed for a whole induced feature set rather than
for independent per-example outputs.

## Evaluation

- Models: Qwen3-4B, Qwen3-14B, Llama-8B, and Gemma3-27B.
- Datasets: Twitter financial news sentiment, Yahoo Finance news sentiment,
  and ToxicChat input-only toxicity classification.
- Setup: `16` training examples per class for prompt grounding, `512`
  annotation examples for feature extraction and evaluation, `16` example
  sets by default, and feature proposal initially constrained to `5-10`
  features.
- Downstream model: logistic regression over extracted feature values.
- Optimization score: F1 combined with an interpretability estimate.

The reported results favor dataset-level textual feedback through
`REFLECTIVEPROPOSER` over unoptimized prompts, scalar-only feedback variants,
and a retrofitted MIPRO-style per-example feedback setup on ToxicChat. The
paper also reports that increasing the number of example sets can improve
performance and stability up to a point, after which the larger search space
can outgrow the available optimization budget.

The experiments were run with vLLM on a `2xA100` cluster, which is relevant
when comparing this approach with SemAxis's simpler fit-time generation plus
NLI scoring design.

## Label-Proxy Concern

The paper explicitly identifies a failure mode where a proposed feature
directly restates the target label and thereby moves the classification task
into the extractor. Its sentiment example is a feature like
`overall_sentimentcategorical`; reported leaky features in the financial-news
experiments include target-like sentiment-label features.

This is nearly the same issue tracked in this repository as label-proxy
leakage. Cosma et al. strengthen the case for auditing SemAxis hypotheses
because they show that optimizing feature discovery for F1 alone can reward
target-restating features. Their solution is an interpretability scorer that
penalizes such features; SemAxis should first measure the problem before
making automatic filtering a default behavior.

## Cost And Scaling

The paper's asymptotic analysis argues that optimization cost is dominated by
running the extractor over the annotation split for each candidate prompt. For
SemAxis, the analogous bottleneck is NLI scoring over text-feature pairs.

This supports reporting feature count, retained feature count, and scoring
work separately. It also reinforces the value of semantic hypothesis
deduplication: fewer retained hypotheses reduce the main measurement cost
before downstream modeling begins.

## Comparison With Current SemAxis

| Dimension | Cosma et al. | SemAxis Today |
|---|---|---|
| Feature discovery | Prompt-optimized LM feature schemas from labeled texts | One-shot LLM hypothesis generation from collection or class contrasts |
| Feature representation | Names, types, descriptions, and extraction prompts | Declarative natural-language hypotheses |
| Feature measurement | LM extractor produces structured values | NLI entailment probability |
| Feedback loop | Dataset-level F1, SHAP, MI, coverage, interpretability | No prompt optimization loop |
| Leakage handling | Interpretability scorer penalizes label-leaking features | Audit issue drafted; no automatic handling yet |
| Downstream model | Logistic regression in experiments | Any sklearn downstream estimator |

## Actionable Findings

1. Promote this paper from `to_review` to `must_read`; it is direct related
   work for supervised feature discovery, not only a candidate prompt
   optimization citation.
2. Strengthen
   [`../ideas/label-proxy-leakage-audit.md`](../ideas/label-proxy-leakage-audit.md):
   Cosma et al. independently observe that F1-only feature discovery can
   produce label-restating features and artificially high performance.
3. Keep automatic prompt optimization out of the current issue scope. It is a
   future research direction after SemAxis has baseline audits for leakage,
   redundancy, stability, and measurement cost.
4. Cite the paper when discussing cost: their extractor bottleneck maps to
   SemAxis's NLI scoring bottleneck over retained hypotheses.

## Issue Candidate Impact

Cosma et al. directly strengthens
[`../ideas/label-proxy-leakage-audit.md`](../ideas/label-proxy-leakage-audit.md).
It also provides secondary support for
[`../ideas/semantic-hypothesis-dedup.md`](../ideas/semantic-hypothesis-dedup.md)
through the shared observation that feature measurement dominates practical
cost, but it does not require expanding the deduplication issue.

## Sources Checked

- arXiv abstract and PDF, accessed 2026-06-07:
  <https://arxiv.org/abs/2601.13922>
- OpenAlex record queried by prior catalog population:
  <https://openalex.org/W7125115128>
