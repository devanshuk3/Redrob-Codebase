"""
Ranker — sorts candidates by final score and selects the top N.
"""

from typing import Any, Dict, List

from src.utils.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def rank_candidates(
    scored_candidates: List[Dict[str, Any]],
    top_n: int = None,
) -> List[Dict[str, Any]]:
    """
    Sort candidates by final_score descending and assign ranks.

    Args:
        scored_candidates: List of dicts with 'candidate_id' and 'final_score'.
        top_n: Number of top candidates to return. Defaults to Config.TOP_N (100).

    Returns:
        Top N candidates with 'rank' field added, sorted by score.
    """
    top_n = top_n or Config.TOP_N

    # Sort by final_score descending, tie-break on candidate_id ascending
    sorted_candidates = sorted(
        scored_candidates,
        key=lambda x: (-x["final_score"], x["candidate_id"]),
    )

    # Select top N
    top = sorted_candidates[:top_n]

    # Assign unique ranks
    for i, candidate in enumerate(top, start=1):
        candidate["rank"] = i

    # Verify uniqueness
    ids = [c["candidate_id"] for c in top]
    ranks = [c["rank"] for c in top]
    assert len(set(ids)) == len(ids), "Duplicate candidate IDs in top N"
    assert len(set(ranks)) == len(ranks), "Duplicate ranks in top N"

    logger.info(
        f"Ranked top {len(top)} candidates "
        f"(score range: {top[0]['final_score']:.4f} – {top[-1]['final_score']:.4f})"
    )

    return top
