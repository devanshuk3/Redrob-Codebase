"""
Semantic ranker — computes cosine similarity between JD and candidate embeddings.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_semantic_scores(
    jd_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_ids: List[str],
) -> Dict[str, float]:
    """
    Compute cosine similarity between the JD embedding and each candidate embedding.

    Args:
        jd_embedding: 1D array of the JD embedding.
        candidate_embeddings: 2D array (n_candidates, embedding_dim).
        candidate_ids: List of candidate IDs matching the embeddings.

    Returns:
        Dictionary mapping candidate_id → semantic_score [0, 1].
    """
    if jd_embedding.ndim == 1:
        jd_embedding = jd_embedding.reshape(1, -1)

    # Compute cosine similarity (batch operation)
    similarities = cosine_similarity(jd_embedding, candidate_embeddings)[0]

    # Normalize to [0, 1] — cosine sim can range from -1 to 1
    # For normalized embeddings, range is typically [0, 1]
    similarities = np.clip(similarities, 0.0, 1.0)

    scores = {}
    for cid, sim in zip(candidate_ids, similarities):
        scores[str(cid)] = round(float(sim), 4)

    logger.info(
        f"Computed semantic scores for {len(scores)} candidates "
        f"(mean={np.mean(similarities):.4f}, max={np.max(similarities):.4f})"
    )

    return scores


def compute_concept_scores(
    concept_embeddings: dict,
    candidate_embeddings: np.ndarray,
    candidate_ids: list,
) -> dict:
    """
    Compute per-concept cosine similarity scores for each candidate.

    Args:
        concept_embeddings: {concept_name: np.ndarray} — 1D embedding per concept.
        candidate_embeddings: 2D array (n_candidates, embedding_dim).
        candidate_ids: List of candidate IDs matching the embeddings.

    Returns:
        {candidate_id: {concept_name: score}} for each concept.
        Scores are raw cosine similarities, clipped to [0, 1].
    """
    result = {}

    for cid in candidate_ids:
        result[str(cid)] = {}

    for concept_name, concept_emb in concept_embeddings.items():
        if concept_emb.ndim == 1:
            concept_emb = concept_emb.reshape(1, -1)

        similarities = cosine_similarity(concept_emb, candidate_embeddings)[0]
        similarities = np.clip(similarities, 0.0, 1.0)

        for cid, sim in zip(candidate_ids, similarities):
            result[str(cid)][concept_name] = round(float(sim), 4)

    logger.info(
        f"Computed concept scores for {len(result)} candidates "
        f"across {len(concept_embeddings)} concepts"
    )

    return result

