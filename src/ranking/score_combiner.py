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
    lexical_scores: Dict[str, float] = None,
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
        lexical = lexical_scores.get(cid, 0.0) if lexical_scores else 0.0
        # 1. Base combination based on weights
        base_score = (
            Config.WEIGHT_SEMANTIC * semantic
            + Config.WEIGHT_STRUCTURED * structured
            + Config.WEIGHT_BEHAVIORAL * behavioral
            + Config.WEIGHT_QUALITY * quality
            + Config.WEIGHT_LEXICAL * lexical
        )

        # 2. Apply experience fit modifier (TASK 1 - smooth dynamic curve: inside -> 1.0, 1 yr -> 0.9, 3 yr -> 0.75, 6+ yr -> 0.5)
        exp_fit = features.get("experience_fit_score", 1.0)
        fit_modifier = exp_fit
        final_score = base_score * fit_modifier

        # 3. Apply trap probability penalty (TASK 3)
        trap_prob = features.get("trap_probability", 0.0)
        trap_penalty = 1.0 - 0.20 * trap_prob
        final_score = final_score * trap_penalty

        # Technical gate: retrieval_score is the non-negotiable minimum
        # No amount of skill/production score compensates for zero retrieval depth
        _retrieval_raw  = features.get("retrieval_score", 0.0)
        _ranking_raw    = features.get("ranking_score", 0.0)
        _evaluation_raw = features.get("evaluation_score", 0.0)
        _skill_raw      = features.get("skill_score", 0.0)
        core_technical = (
            _retrieval_raw * 0.40
            + _ranking_raw * 0.30
            + _evaluation_raw * 0.20
            + _skill_raw * 0.10
        )
        if _retrieval_raw < 0.20:
            final_score *= 0.20
        elif _retrieval_raw < 0.30:
            final_score *= 0.40
        elif core_technical < 0.15:
            final_score *= 0.35
        elif core_technical < 0.25:
            final_score *= 0.65

        # Evaluation boost: reward genuine evaluation experience (NDCG/MRR/A-B testing)
        if _evaluation_raw >= 0.40:
            final_score *= 1.20
        elif _evaluation_raw >= 0.20:
            final_score *= 1.12
        elif _evaluation_raw >= 0.10:
            final_score *= 1.05

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
        
        # 5. Apply location and availability multipliers (to penalize unresponsive or international candidates)
        availability_mult = 1.0
        
        last_active_date = features.get("last_active_date", "")
        days_inactive = 0
        if last_active_date:
            try:
                parts = last_active_date.split("-")
                if len(parts) == 3:
                    y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                    from datetime import date
                    ref_date = max(date.today(), date(2026, 6, 18))
                    days_inactive = (ref_date - date(y, m, d)).days
            except:
                pass
                
        open_to_work = features.get("open_to_work", True)
        if not open_to_work:
            availability_mult *= 0.85
            
        resp_rate = features.get("recruiter_response_rate", 1.0)
        if resp_rate < 0.10:
            availability_mult *= 0.70
        elif resp_rate < 0.20:
            availability_mult *= 0.85
            
        if days_inactive > 180:
            availability_mult *= 0.70
        elif days_inactive > 120:
            availability_mult *= 0.85
        elif days_inactive > 90:
            availability_mult *= 0.95
            
        if days_inactive > 150 and resp_rate < 0.15:
            availability_mult *= 0.80

        notice_days = features.get("notice_period_days", 0)
        if notice_days > 90:
            availability_mult *= 0.75
        elif notice_days > 60:
            availability_mult *= 0.88

        location_mult = 1.0
        country = (features.get("country") or "").strip().lower()
        willing_to_relocate = features.get("willing_to_relocate", True)
        city_loc = (features.get("location") or "").strip().lower()
        
        if country and country != "india":
            if not willing_to_relocate:
                location_mult = 0.0
            else:
                location_mult = 0.20
        else:
            is_preferred_city = any(city in city_loc for city in ["pune", "noida", "delhi", "ncr", "gurgaon", "ghaziabad", "faridabad"])
            if not is_preferred_city and not willing_to_relocate:
                location_mult = 0.50
                
        # Consulting penalty: penalize consulting-heavy careers (TCS/Infosys/Wipro/etc.)
        consulting_ratio = features.get("consulting_ratio", 0.0)
        consulting_mult = 1.0
        if consulting_ratio >= 1.0:
            consulting_mult = 0.40
        elif consulting_ratio >= 0.75:
            consulting_mult = 0.50
        elif consulting_ratio >= 0.50:
            consulting_mult = 0.60
        elif consulting_ratio >= 0.25:
            consulting_mult = 0.80

        final_score = final_score * domain_boost * scale_boost * behavioral_boost * availability_mult * location_mult * consulting_mult

        entry = {
            "candidate_id": cid,
            "final_score": round(final_score, 6),
            "semantic_score": semantic,
            "lexical_score": lexical,
            "structured_score": structured,
            "behavioral_score": behavioral,
            "quality_score": quality,
            "availability_mult": round(availability_mult, 4),
            "location_mult": round(location_mult, 4),
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
