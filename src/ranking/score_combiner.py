"""
Score combiner — aggregates semantic, structured, behavioral, and quality scores
into a final candidate score (TASK 11 — rebalanced for technical dominance, TASK 1 & 3 modifiers).

Weights:
  0.20 semantic + 0.55 structured + 0.15 behavioral + 0.10 quality

Technical relevance dominates. Behavioral signals modify fit, not define it.
Refined with experience fit and trap probability modifiers.
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

        # 1. Base combination based on weights
        base_score = (
            Config.WEIGHT_SEMANTIC * semantic
            + Config.WEIGHT_STRUCTURED * structured
            + Config.WEIGHT_BEHAVIORAL * behavioral
            + Config.WEIGHT_QUALITY * quality
        )

        # 2. Apply experience fit modifier (TASK 1)
        exp_fit = features.get("experience_fit_score", 1.0)
        fit_modifier = 0.70 + 0.30 * exp_fit
        final_score = base_score * fit_modifier

        # 3. Apply trap probability penalty (TASK 3)
        trap_prob = features.get("trap_probability", 0.0)
        trap_penalty = 1.0 - 0.20 * trap_prob
        final_score = final_score * trap_penalty

        results.append({
            "candidate_id": cid,
            "final_score": round(final_score, 6),
            "semantic_score": semantic,
            "structured_score": structured,
            "behavioral_score": behavioral,
            "quality_score": quality,
            # Pass through all sub-scores for reasoning and debug
            **{k: v for k, v in features.items() if k not in ("candidate_id", "structured_score", "behavioral_score")},
        })

    logger.info(f"Combined scores for {len(results)} candidates")
    return results
