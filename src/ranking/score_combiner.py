"""
Score combiner — aggregates semantic, structured, behavioral, and quality scores
into a final candidate score (CHANGE 7 — rebalanced weights).

Default weights:
  0.25 semantic + 0.45 structured + 0.20 behavioral + 0.10 quality
"""

from typing import Any, Dict, List

from src.utils.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def combine_scores(
    candidate_features: Dict[str, Dict[str, Any]],
    semantic_scores: Dict[str, float],
    quality_scores: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Combine all score dimensions into a final ranking score.

    Args:
        candidate_features: {candidate_id: {structured_score, behavioral_score, ...}}
        semantic_scores: {candidate_id: semantic_score}
        quality_scores: {candidate_id: quality_score}

    Returns:
        List of dicts with candidate_id, final_score, and all sub-scores.
    """
    results = []

    for cid, features in candidate_features.items():
        semantic = semantic_scores.get(cid, 0.0)
        quality = quality_scores.get(cid, 0.5)
        structured = features.get("structured_score", 0.0)
        behavioral = features.get("behavioral_score", 0.0)

        final_score = (
            Config.WEIGHT_SEMANTIC * semantic
            + Config.WEIGHT_STRUCTURED * structured
            + Config.WEIGHT_BEHAVIORAL * behavioral
            + Config.WEIGHT_QUALITY * quality
        )

        results.append({
            "candidate_id": cid,
            "final_score": round(final_score, 6),
            "semantic_score": semantic,
            "structured_score": structured,
            "behavioral_score": behavioral,
            "quality_score": quality,
            # Pass through all sub-scores for reasoning
            **{k: v for k, v in features.items() if k not in ("candidate_id", "structured_score", "behavioral_score")},
        })

    logger.info(f"Combined scores for {len(results)} candidates")
    return results
