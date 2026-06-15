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
    concept_scores: Dict[str, Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Combine all score dimensions into a final ranking score.

    Args:
        candidate_features: {candidate_id: {structured_score, behavioral_score, ...}}
        semantic_scores: {candidate_id: semantic_score}
        quality_scores: {candidate_id: quality_score}
        concept_scores: Optional {candidate_id: {concept_name: score}} from concept embeddings.

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

        # 2. Apply experience fit modifier (TASK 1 - smooth dynamic curve: inside -> 1.0, 1 yr -> 0.9, 3 yr -> 0.75, 6+ yr -> 0.5)
        exp_fit = features.get("experience_fit_score", 1.0)
        fit_modifier = exp_fit
        final_score = base_score * fit_modifier

        # 3. Apply trap probability penalty (TASK 3)
        trap_prob = features.get("trap_probability", 0.0)
        trap_penalty = 1.0 - 0.20 * trap_prob
        final_score = final_score * trap_penalty

        # 4. Apply positive technical rewards (TASK 6 - shift to 70% reward, 30% penalty philosophy)
        retrieval = features.get("retrieval_score", 0.0)
        ranking = features.get("ranking_score", 0.0)
        evaluation = features.get("evaluation_score", 0.0)
        production = features.get("production_score", 0.0)

        # Blend with concept scores if available (TASK C)
        cid_concepts = {}
        if concept_scores and cid in concept_scores:
            cid_concepts = concept_scores[cid]
            effective_retrieval  = 0.6 * retrieval  + 0.4 * cid_concepts.get("ranking", 0)
            effective_ranking    = 0.6 * ranking    + 0.4 * cid_concepts.get("ranking", 0)
            effective_evaluation = 0.6 * evaluation + 0.4 * cid_concepts.get("evaluation", 0)
            effective_production = 0.6 * production + 0.4 * cid_concepts.get("production", 0)
        else:
            effective_retrieval = retrieval
            effective_ranking = ranking
            effective_evaluation = evaluation
            effective_production = production

        tech_scores = [effective_retrieval, effective_ranking, effective_evaluation, effective_production]
        strong_tech_count = sum(1 for s in tech_scores if s >= 0.6)
        domain_boost = 1.0
        if strong_tech_count >= 3:
            domain_boost = 1.05  # 5% boost for true multi-domain technical experts
        elif strong_tech_count == 2:
            domain_boost = 1.02  # 2% boost
            
        scale_boost = features.get("scale_boost", 1.0)
        behavioral_boost = features.get("behavioral_boost", 1.0)
        
        final_score = final_score * domain_boost * scale_boost * behavioral_boost

        entry = {
            "candidate_id": cid,
            "final_score": round(final_score, 6),
            "semantic_score": semantic,
            "structured_score": structured,
            "behavioral_score": behavioral,
            "quality_score": quality,
            # Pass through all sub-scores for reasoning and debug
            **{k: v for k, v in features.items() if k not in ("candidate_id", "structured_score", "behavioral_score")},
        }

        # Include concept scores in output for debug visibility
        if cid_concepts:
            entry["ranking_concept"] = cid_concepts.get("ranking", 0.0)
            entry["evaluation_concept"] = cid_concepts.get("evaluation", 0.0)
            entry["production_concept"] = cid_concepts.get("production", 0.0)

        results.append(entry)

    logger.info(f"Combined scores for {len(results)} candidates")
    return results
