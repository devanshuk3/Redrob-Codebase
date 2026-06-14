"""
Skill extraction and matching against JD requirements (TASK 1, 2, 8).
Uses exact + fuzzy matching (rapidfuzz) with rich metadata exploitation:
  - proficiency weighting
  - duration weighting (log scale)
  - endorsement bonus
  - skill assessment score integration
"""

import math
from typing import Dict, List, Optional

from rapidfuzz import fuzz

from src.ingestion.candidate_parser import Candidate
from src.features.jd_feature_mapper import JDFeatures
from src.utils.text_utils import canonicalize_skill
from src.utils.constants import PROFICIENCY_WEIGHTS
from src.utils.config import Config


def _proficiency_weight(proficiency: str) -> float:
    """Map proficiency string to weight (TASK 2 — exploit metadata)."""
    return PROFICIENCY_WEIGHTS.get(proficiency.lower(), 0.4)


def _duration_weight(duration_months: int) -> float:
    """
    Compute duration-based weight using log scale (TASK 2).
    Long-term advanced skills outrank short-term beginner skills.
    log(duration_months + 1) normalized to [0.3, 1.0].
    """
    if duration_months <= 0:
        return 0.5  # unknown duration → neutral
    raw = math.log(duration_months + 1)
    # log(1) = 0, log(13) ≈ 2.56, log(61) ≈ 4.11, log(121) ≈ 4.80
    # Normalize: 12 months → ~0.6, 60 months → ~0.9, 120+ → 1.0
    normalized = min(1.0, 0.3 + raw * 0.15)
    return normalized


def _endorsement_bonus(endorsements: int) -> float:
    """Mild endorsement bonus (TASK 2). Capped at +0.1."""
    if endorsements <= 0:
        return 0.0
    # Diminishing returns: 10 endorsements → ~0.05, 50+ → ~0.1
    return min(0.1, math.log(endorsements + 1) * 0.025)


def _assessment_modifier(
    skill_name: str,
    assessment_scores: Dict[str, float],
) -> float:
    """
    Use skill_assessment_scores to modify confidence (TASK 8).
    High scores (80+) boost, low scores (<50) reduce confidence.
    Returns a multiplier in [0.7, 1.15].
    """
    if not assessment_scores:
        return 1.0

    # Try exact match first
    skill_lower = skill_name.lower()
    for key, score_val in assessment_scores.items():
        if key.lower() == skill_lower or skill_lower in key.lower() or key.lower() in skill_lower:
            try:
                score_val = float(score_val)
            except (ValueError, TypeError):
                continue
            if score_val >= 80:
                return 1.15
            elif score_val >= 60:
                return 1.0
            elif score_val >= 40:
                return 0.85
            else:
                return 0.7
    return 1.0


def compute_skill_strength(
    skill_entry,
    assessment_scores: Dict[str, float] = None,
) -> float:
    """
    Compute per-skill strength (TASK 2 — exploit all metadata):
    skill_strength = proficiency_weight × duration_weight × endorsement_bonus × assessment_modifier

    Returns a value typically in [0.1, 1.3].
    """
    prof = _proficiency_weight(skill_entry.proficiency)
    dur = _duration_weight(skill_entry.duration_months)
    endorse = _endorsement_bonus(skill_entry.endorsements)
    assess = _assessment_modifier(
        skill_entry.name,
        assessment_scores or {},
    )

    # Base strength = proficiency * duration, with endorsement added
    strength = (prof * dur + endorse) * assess
    return strength


def score_skills(candidate: Candidate, jd_features: JDFeatures) -> float:
    """
    Score candidate skills against JD requirements (TASK 1 — fixed, TASK 2 — enriched).

    Uses fuzzy matching with full metadata exploitation:
    - proficiency × duration × endorsements × assessment scores.
    Must-have skills are weighted more heavily than preferred.

    Returns:
        Normalized score [0, 1].
    """
    if not jd_features.all_skills():
        return 0.0

    candidate_skills = {
        canonicalize_skill(s.name): s for s in candidate.skills if s.name
    }

    if not candidate_skills:
        return 0.0

    assessment_scores = getattr(
        candidate.redrob_signals, 'skill_assessment_scores', {}
    ) or {}

    must_have = jd_features.must_have_skills
    preferred = jd_features.preferred_skills

    must_have_score = _match_skill_list(candidate_skills, must_have, assessment_scores)
    preferred_score = _match_skill_list(candidate_skills, preferred, assessment_scores)

    # Must-have skills are 70% of the skill score, preferred 30%
    if must_have and preferred:
        score = 0.70 * must_have_score + 0.30 * preferred_score
    elif must_have:
        score = must_have_score
    else:
        score = preferred_score

    return round(min(1.0, score), 4)


def _match_skill_list(
    candidate_skills: Dict,
    target_skills: List[str],
    assessment_scores: Dict[str, float] = None,
) -> float:
    """
    Compute match ratio for a list of target skills.
    Uses exact match first, then fuzzy fallback.
    Now uses full skill metadata for richer scoring.
    """
    if not target_skills:
        return 0.0

    total_weight = 0.0
    matched_weight = 0.0

    for target in target_skills:
        target_canon = canonicalize_skill(target)
        total_weight += 1.0

        # Exact match
        if target_canon in candidate_skills:
            skill_entry = candidate_skills[target_canon]
            strength = compute_skill_strength(skill_entry, assessment_scores)
            matched_weight += min(1.0, strength)
            continue

        # Fuzzy match
        best_score = 0
        best_skill_entry = None
        for cand_skill, skill_entry in candidate_skills.items():
            ratio = fuzz.ratio(target_canon, cand_skill)
            if ratio > best_score:
                best_score = ratio
                best_skill_entry = skill_entry

        if best_score >= Config.FUZZY_MATCH_THRESHOLD and best_skill_entry:
            strength = compute_skill_strength(best_skill_entry, assessment_scores)
            # Discount for fuzzy match
            fuzzy_factor = best_score / 100.0
            matched_weight += min(1.0, strength * fuzzy_factor)

    return matched_weight / total_weight if total_weight > 0 else 0.0


def get_matched_skills(candidate: Candidate, jd_features: JDFeatures) -> List[str]:
    """Return list of skill names that matched the JD (for reasoning). Deduplicated."""
    candidate_skills = {
        canonicalize_skill(s.name): s.name for s in candidate.skills if s.name
    }
    matched = []
    seen = set()
    for target in jd_features.all_skills():
        target_canon = canonicalize_skill(target)
        if target_canon in candidate_skills:
            name = candidate_skills[target_canon]
            if name.lower() not in seen:
                seen.add(name.lower())
                matched.append(name)
            continue
        for cand_skill, orig_name in candidate_skills.items():
            if fuzz.ratio(target_canon, cand_skill) >= Config.FUZZY_MATCH_THRESHOLD:
                if orig_name.lower() not in seen:
                    seen.add(orig_name.lower())
                    matched.append(orig_name)
                break
    return matched
