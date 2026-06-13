"""
Skill extraction and matching against JD requirements.
Uses exact + fuzzy matching (rapidfuzz).
"""

from typing import Dict, List

from rapidfuzz import fuzz

from src.ingestion.candidate_parser import Candidate
from src.features.jd_feature_mapper import JDFeatures
from src.utils.text_utils import canonicalize_skill
from src.utils.constants import PROFICIENCY_WEIGHTS
from src.utils.config import Config


def score_skills(candidate: Candidate, jd_features: JDFeatures) -> float:
    """
    Score candidate skills against JD requirements.

    Uses fuzzy matching with proficiency weighting.
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

    must_have = jd_features.must_have_skills
    preferred = jd_features.preferred_skills

    must_have_score = _match_skill_list(candidate_skills, must_have)
    preferred_score = _match_skill_list(candidate_skills, preferred)

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
) -> float:
    """
    Compute match ratio for a list of target skills.
    Uses exact match first, then fuzzy fallback.
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
            prof_weight = PROFICIENCY_WEIGHTS.get(skill_entry.proficiency, 0.3)
            matched_weight += prof_weight
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
            prof_weight = PROFICIENCY_WEIGHTS.get(best_skill_entry.proficiency, 0.3)
            # Discount slightly for fuzzy match
            matched_weight += prof_weight * (best_score / 100.0)

    return matched_weight / total_weight if total_weight > 0 else 0.0


def get_matched_skills(candidate: Candidate, jd_features: JDFeatures) -> List[str]:
    """Return list of skill names that matched the JD (for reasoning)."""
    candidate_skills = {
        canonicalize_skill(s.name): s.name for s in candidate.skills if s.name
    }
    matched = []
    for target in jd_features.all_skills():
        target_canon = canonicalize_skill(target)
        if target_canon in candidate_skills:
            matched.append(candidate_skills[target_canon])
            continue
        for cand_skill, orig_name in candidate_skills.items():
            if fuzz.ratio(target_canon, cand_skill) >= Config.FUZZY_MATCH_THRESHOLD:
                matched.append(orig_name)
                break
    return matched
