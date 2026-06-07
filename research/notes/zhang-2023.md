# Zhang et al. (2023)

## Citation

Shaokun Zhang, Xiaobo Xia, Zhaoqing Wang, Ling-Hao Chen, Jiale Liu, Qingyun
Wu, and Tongliang Liu. 2023. *IDEAL: Influence-Driven Selective Annotations
Empower In-Context Learners in Large Language Models*. arXiv:2310.10873;
published as an ICLR 2024 conference paper. DOI:
[`10.48550/arXiv.2310.10873`](https://doi.org/10.48550/arXiv.2310.10873).

- OpenReview: `Spp2i1hKwV`
- OpenAlex: `W4387806062`
- Local PDF cache: `../pdfs/zhang-2023-ideal.pdf`
- arXiv record: <https://arxiv.org/abs/2310.10873>
- OpenReview page: <https://openreview.net/forum?id=Spp2i1hKwV>
- Project page: <https://skzhang1.github.io/IDEAL/>

## Why It Matters For SemAxis

IDEAL is a useful follow-up to Vote-K. Su et al. support Vote-K as a strong
selective annotation method, but Zhang et al. argue that Vote-K is not
end-to-end, requires a delicate diversity/representativeness balance, can
select outliers, and lacks theoretical guarantees. IDEAL replaces that balance
with an influence-maximization objective over a directed similarity graph.

For SemAxis, this does not mean IDEAL should immediately replace Vote-K.
IDEAL, like Vote-K, was designed for selective annotation and per-test prompt
retrieval, not for fit-time hypothesis generation. It does mean SemAxis should
benchmark sampling methods instead of treating any one method as settled.

The paper is especially relevant because SemAxis already exposes `random`,
`kmeans`, and `votek`. A next research issue should compare these strategies
against influence- or coverage-based alternatives under the same prompt budget.

## Method

IDEAL constructs a directed graph over unlabeled examples:

1. Embed each unlabeled example with Sentence-BERT.
2. Connect each vertex to its `k` nearest successors by cosine similarity.
3. Weight outgoing edges by normalized cosine similarity.
4. Quantify a candidate subset's influence using an independent-cascade
   diffusion process on the graph.
5. Greedily select the next example that maximizes marginal gain in subset
   influence until the annotation budget is reached.

The method is inspired by influence maximization. Under a submodularity
condition, the greedy search has the familiar `1 - 1/e` style approximation
guarantee as the annotation budget grows.

After subset selection, the paper still uses prompt retrieval from the selected
annotated pool. IDEAL is therefore a replacement for the selective-annotation
step, not a replacement for per-test retrieval.

## Evaluation

- Datasets: MRPC, SST-5, MNLI, DBpedia, RTE, HellaSwag, MultiWOZ, GeoQuery,
  and XSum.
- Tasks: classification, multiple-choice, dialogue state tracking, semantic
  parsing, and summarization.
- Models: GPT-J and text-davinci-002 by default, with GPT-Neo and
  GPT-3.5-Turbo explored in analysis.
- Subsampling: selective annotation is performed from `3K` randomly sampled
  training examples, repeated across trials.
- Graph setting: `k = 10` nearest successors in main experiments.

The paper reports that IDEAL outperforms Vote-K in `17` out of `18` main
comparison cases and uses about `13%` of Vote-K's subset-selection time. It
also reports better stability under prompt-order permutations and includes UMAP
visualizations where IDEAL-selected examples better cover the data manifold
than Vote-K-selected examples.

## Limitations Relevant To SemAxis

The same transfer caveat applies as with Vote-K: IDEAL selects examples to
annotate for later prompt retrieval. SemAxis selects examples to condition a
hypothesis generator during `fit`. The objective may be useful, but the paper's
accuracy gains are not direct evidence for better SemAxis hypotheses.

IDEAL also adds implementation complexity: graph construction, stochastic
diffusion simulation, repeated influence estimation, and greedy marginal-gain
search. For SemAxis, a simpler facility-location or deterministic coverage
baseline may be a more practical first addition.

The paper's critique of Vote-K is still important. Diversity-heavy selection
can overselect outliers, and selection methods should be compared on stability,
feature redundancy, label-proxy rate, and NLI cost, not only downstream
accuracy.

## Comparison With Current SemAxis

| Dimension | IDEAL | SemAxis Today |
|---|---|---|
| Selection objective | Maximize influence over similarity graph | Random, K-Means, or Vote-K |
| Original use | Select examples to annotate for ICL prompt retrieval | Select examples for hypothesis generation prompts |
| Graph parameters | Directed `k`-NN graph, diffusion probabilities | Vote-K graph parameters where applicable |
| Strength | Theoretical support and efficient selection vs Vote-K | Simple implemented baselines |
| Transfer caveat | Still assumes later prompt retrieval | No per-test retrieval for generated features |

## Actionable Findings

1. Create a sampling-strategy benchmark candidate for SemAxis rather than
   replacing Vote-K immediately.
2. Compare existing `random`, `kmeans`, and `votek` methods against at least
   one coverage/influence-oriented alternative under fixed token budgets.
3. Track selection diagnostics: selected row ids, embedding model, graph
   parameters, prompt order, pairwise similarity, coverage, and outlier scores.
4. Evaluate generated hypotheses through redundancy, label-proxy rate,
   stability, downstream metric, and NLI pair count.
5. Treat IDEAL as a research baseline or inspiration; do not expand core
   implementation before a smaller benchmark justifies the complexity.

## Issue Candidate Impact

This paper motivates
[`../ideas/sampling-strategy-benchmark.md`](../ideas/sampling-strategy-benchmark.md).
It also strengthens
[`../ideas/experiment-audit-logging.md`](../ideas/experiment-audit-logging.md),
because selection diagnostics and graph parameters must be saved to compare
sampling strategies rigorously.

## Sources Checked

- OpenReview PDF and page, accessed 2026-06-07:
  <https://openreview.net/forum?id=Spp2i1hKwV>
- arXiv abstract, accessed 2026-06-07:
  <https://arxiv.org/abs/2310.10873>
- OpenAlex record added through `uvx littrail add-paper`, accessed 2026-06-07:
  <https://openalex.org/W4387806062>
