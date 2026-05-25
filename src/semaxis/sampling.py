from __future__ import annotations

import random
from typing import Callable

import numpy as np


def sample_texts_within_budget(
    texts: list[str],
    token_budget: int,
    tokenizer_fn: Callable[[str], int],
    rng: random.Random | None = None,
) -> list[str]:
    """Return a random subset of texts that fits within the token budget.

    Texts are shuffled, then added one by one until adding the next text
    would exceed token_budget. The order of the returned list follows the
    shuffled order.

    Args:
        texts: Source texts to sample from.
        token_budget: Maximum total tokens allowed.
        tokenizer_fn: Function that returns the token count for a single text.
        rng: Optional Random instance for reproducibility.

    Returns:
        A list of texts whose total token count does not exceed token_budget.
    """
    if rng is None:
        rng = random.Random()

    indices = list(range(len(texts)))
    rng.shuffle(indices)

    selected: list[str] = []
    total_tokens = 0

    for idx in indices:
        text = texts[idx]
        count = tokenizer_fn(text)
        if total_tokens + count > token_budget:
            break
        selected.append(text)
        total_tokens += count

    return selected


def sample_texts_kmeans(
    texts: list[str],
    n: int,
    embeddings: np.ndarray,
    rng: random.Random | None = None,
) -> list[str]:
    """Return up to n texts by selecting the closest text to each K-Means centroid.

    Args:
        texts: Source texts.
        n: Number of texts to select (capped at len(texts)).
        embeddings: Pre-computed embedding matrix of shape (len(texts), dim).
        rng: Optional Random instance for reproducibility.

    Returns:
        A list of up to n representative texts, one per cluster.
    """
    from sklearn.cluster import KMeans

    n = min(n, len(texts))
    if n == 0:
        return []
    if n == len(texts):
        return list(texts)

    seed = rng.randint(0, 2**31 - 1) if rng is not None else None
    km = KMeans(n_clusters=n, random_state=seed, n_init="auto")
    km.fit(embeddings)

    selected_indices: list[int] = []
    for center in km.cluster_centers_:
        dists = np.linalg.norm(embeddings - center, axis=1)
        # Exclude already-selected indices to avoid duplicates when clusters share a medoid
        dists[selected_indices] = np.inf
        selected_indices.append(int(np.argmin(dists)))

    return [texts[i] for i in selected_indices]


def sample_texts_votek(
    texts: list[str],
    n: int,
    embeddings: np.ndarray,
    k: int = 10,
    rng: random.Random | None = None,
) -> list[str]:
    """Return up to n texts using the Vote-K algorithm (Su et al. 2022).

    Vote-K balances representativeness (high vote count = many neighbours)
    and diversity (selected texts' neighbours are suppressed from future picks).

    Args:
        texts: Source texts.
        n: Number of texts to select (capped at len(texts)).
        embeddings: Pre-computed embedding matrix of shape (len(texts), dim).
        k: Number of neighbours to consider for voting and suppression.
        rng: Optional Random instance (unused; kept for API symmetry).

    Returns:
        A list of up to n texts.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    n = min(n, len(texts))
    if n == 0:
        return []
    if n == len(texts):
        return list(texts)

    k = min(k, len(texts) - 1)
    sim = cosine_similarity(embeddings)  # (N, N)

    # votes[i] = how many texts consider i among their top-k neighbours
    votes = np.zeros(len(texts), dtype=float)
    for i in range(len(texts)):
        sim_row = sim[i].copy()
        sim_row[i] = -np.inf  # exclude self
        top_k = np.argpartition(sim_row, -k)[-k:]
        votes[top_k] += 1.0

    selected_indices: list[int] = []
    remaining = np.ones(len(texts), dtype=bool)

    while len(selected_indices) < n and remaining.any():
        # Among remaining texts, pick the one with the most votes
        masked_votes = np.where(remaining, votes, -np.inf)
        best = int(np.argmax(masked_votes))
        selected_indices.append(best)
        remaining[best] = False

        # Suppress votes of k nearest neighbours
        sim_row = sim[best].copy()
        sim_row[best] = -np.inf
        top_k = np.argpartition(sim_row, -k)[-k:]
        votes[top_k] = 0.0

    return [texts[i] for i in selected_indices]


def _estimate_n(
    texts: list[str],
    token_budget: int,
    tokenizer_fn: Callable[[str], int],
) -> int:
    """Estimate how many texts fit in token_budget based on average token length."""
    if not texts:
        return 0
    probe = texts[:min(20, len(texts))]
    avg = sum(tokenizer_fn(t) for t in probe) / len(probe)
    return max(1, int(token_budget / avg))


def _trim_to_budget(
    texts: list[str],
    token_budget: int,
    tokenizer_fn: Callable[[str], int],
) -> list[str]:
    """Trim a list of texts to fit within the token budget."""
    result: list[str] = []
    total = 0
    for t in texts:
        count = tokenizer_fn(t)
        if total + count > token_budget:
            break
        result.append(t)
        total += count
    return result
