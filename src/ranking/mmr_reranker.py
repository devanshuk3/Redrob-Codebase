"""
MMR (Maximal Marginal Relevance) Reranker — lightweight diversity stage.

Inserted after all scoring/filtering and before final top-100 selection.
Controlled by ENABLE_MMR flag; when False the pipeline is unchanged.

MMR formula:
    mmr_score = lambda_ * norm_relevance - (1 - lambda_) * max_sim_to_selected

This module introduces NO new external dependencies.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Feature flag ────────────────────────────────────────────────────────────
ENABLE_MMR = False

# ── Hyperparameters ─────────────────────────────────────────────────────────
MMR_LAMBDA = 0.85
MMR_POOL_SIZE = 150
MMR_SELECT_N = 100


def _build_embedding_index(
    candidate_embeddings: np.ndarray,
    candidate_ids: List[str],
) -> Dict[str, np.ndarray]:
    """Map candidate IDs to their embedding vectors."""
    return {cid: candidate_embeddings[i] for i, cid in enumerate(candidate_ids)}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0.0:
        return 0.0
    return float(dot / norm)


def mmr_rerank(
    scored_candidates: List[Dict[str, Any]],
    candidate_embeddings: Optional[np.ndarray],
    candidate_ids: Optional[List[str]],
    lambda_: float = MMR_LAMBDA,
    pool_size: int = MMR_POOL_SIZE,
    select_n: int = MMR_SELECT_N,
) -> List[Dict[str, Any]]:
    """
    Apply Maximal Marginal Relevance reranking.

    Args:
        scored_candidates: Full list of scored candidate dicts (must have 'final_score' and 'candidate_id').
        candidate_embeddings: numpy array of shape (N, dim) aligned with candidate_ids.
        candidate_ids: list of candidate IDs matching the embedding matrix rows.
        lambda_: Trade-off parameter (higher = more relevance, lower = more diversity).
        pool_size: Number of top candidates to consider for reranking.
        select_n: Number of candidates to select via MMR.

    Returns:
        Reranked list of candidate dicts (length = select_n or fewer).
        Original 'final_score' values are preserved; only ordering changes.
    """
    if not ENABLE_MMR:
        return scored_candidates

    # Guard: need embeddings
    if candidate_embeddings is None or candidate_ids is None:
        logger.warning("MMR skipped — embeddings unavailable")
        return scored_candidates

    try:
        return _mmr_select(
            scored_candidates, candidate_embeddings, candidate_ids,
            lambda_, pool_size, select_n,
        )
    except Exception:
        logger.exception("MMR reranking failed — falling back to original ranking")
        return scored_candidates


def _mmr_select(
    scored_candidates: List[Dict[str, Any]],
    candidate_embeddings: np.ndarray,
    candidate_ids: List[str],
    lambda_: float,
    pool_size: int,
    select_n: int,
) -> List[Dict[str, Any]]:
    """Core MMR selection loop (no exception handling — caller wraps)."""

    # 1. Sort by existing final_score descending and take the pool
    pool = sorted(scored_candidates, key=lambda x: -x["final_score"])[:pool_size]

    if len(pool) <= select_n:
        logger.info(f"MMR pool ({len(pool)}) <= select_n ({select_n}); skipping MMR")
        return pool

    # 2. Build embedding lookup from the full embedding matrix
    emb_index = _build_embedding_index(candidate_embeddings, candidate_ids)

    # Filter pool to candidates that have embeddings
    pool_with_emb = [c for c in pool if c["candidate_id"] in emb_index]
    pool_without_emb = [c for c in pool if c["candidate_id"] not in emb_index]

    if len(pool_with_emb) < select_n:
        logger.warning(
            f"Only {len(pool_with_emb)} pool candidates have embeddings "
            f"(need {select_n}); skipping MMR"
        )
        return scored_candidates

    # 3. Normalize relevance scores within the pool
    scores = np.array([c["final_score"] for c in pool_with_emb], dtype=np.float64)
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min > 1e-12:
        norm_scores = (scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.ones_like(scores)

    # Pre-fetch embeddings for pool candidates (ordered)
    pool_embs = np.stack([emb_index[c["candidate_id"]] for c in pool_with_emb])

    # Pre-compute norms for fast cosine similarity via dot products
    pool_norms = np.linalg.norm(pool_embs, axis=1, keepdims=True)
    pool_norms = np.where(pool_norms == 0, 1.0, pool_norms)
    pool_embs_normed = pool_embs / pool_norms  # (pool_size, dim)

    # 4. Greedy MMR selection
    selected_indices: List[int] = []
    remaining = set(range(len(pool_with_emb)))

    # Select the highest-scoring candidate first
    first_idx = int(np.argmax(norm_scores))
    selected_indices.append(first_idx)
    remaining.discard(first_idx)

    # Track selected embeddings (normed) for fast max-similarity
    selected_embs_normed = [pool_embs_normed[first_idx]]

    for _ in range(select_n - 1):
        if not remaining:
            break

        remaining_list = list(remaining)

        # Compute max cosine similarity of each remaining candidate to selected set
        # selected_matrix: (num_selected, dim)
        selected_matrix = np.stack(selected_embs_normed)  # (k, dim)
        remaining_embs = pool_embs_normed[remaining_list]  # (r, dim)

        # sim_matrix: (r, k)
        sim_matrix = remaining_embs @ selected_matrix.T
        max_sim = sim_matrix.max(axis=1)  # (r,)

        # MMR score for each remaining candidate
        remaining_norm_scores = norm_scores[remaining_list]
        mmr_scores = lambda_ * remaining_norm_scores - (1 - lambda_) * max_sim

        best_remaining_pos = int(np.argmax(mmr_scores))
        best_idx = remaining_list[best_remaining_pos]

        selected_indices.append(best_idx)
        remaining.discard(best_idx)
        selected_embs_normed.append(pool_embs_normed[best_idx])

    # 5. Build result preserving original scores
    result = [pool_with_emb[i] for i in selected_indices]

    logger.info(
        f"MMR reranking complete: selected {len(result)} from pool of {len(pool_with_emb)} "
        f"(lambda={lambda_})"
    )

    return result
