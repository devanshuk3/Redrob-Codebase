"""
Reasoning generator — template-based, no LLM (CHANGE 9).

Generates 1-2 sentence reasoning using only actual candidate data.
No hallucinations. References real skills, experience, and signals.
"""

from typing import Any, Dict, List

from src.ingestion.candidate_parser import Candidate
from src.features.skill_extractor import get_matched_skills
from src.features.jd_feature_mapper import JDFeatures
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def generate_reasoning(
    ranked_candidates: List[Dict[str, Any]],
    candidate_map: Dict[str, Candidate],
    jd_features: JDFeatures,
) -> List[Dict[str, Any]]:
    """
    Generate reasoning strings for each ranked candidate.

    Uses template-based generation referencing actual candidate data:
    - Years of experience
    - Matched skills
    - Career highlights
    - Behavioral strengths

    Args:
        ranked_candidates: List of ranked candidate dicts.
        candidate_map: {candidate_id: Candidate} lookup.
        jd_features: Parsed JD features for skill matching.

    Returns:
        Same list with 'reasoning' field added.
    """
    for entry in ranked_candidates:
        cid = entry["candidate_id"]
        candidate = candidate_map.get(cid)

        if not candidate:
            entry["reasoning"] = "Candidate data not available."
            continue

        entry["reasoning"] = _build_reasoning(entry, candidate, jd_features)

    logger.info(f"Generated reasoning for {len(ranked_candidates)} candidates")
    return ranked_candidates


def _build_reasoning(
    scores: Dict[str, Any],
    candidate: Candidate,
    jd_features: JDFeatures,
) -> str:
    """Build a reasoning string from actual candidate data."""
    parts = []

    # ── Experience ───────────────────────────────────────────────────
    yoe = candidate.years_of_experience
    title = candidate.current_title or "professional"
    company = candidate.current_company

    if company:
        parts.append(f"{yoe:.0f} years of experience as {title} at {company}")
    else:
        parts.append(f"{yoe:.0f} years of experience as {title}")

    # ── Key matched skills ───────────────────────────────────────────
    matched = get_matched_skills(candidate, jd_features)
    if matched:
        top_skills = matched[:5]
        parts.append(f"with skills in {', '.join(top_skills)}")

    # ── Domain strengths ─────────────────────────────────────────────
    strengths = []

    retrieval = scores.get("retrieval_score", 0)
    ranking = scores.get("ranking_score", 0)
    production = scores.get("production_score", 0)
    evaluation = scores.get("evaluation_score", 0)

    if retrieval >= 0.5:
        strengths.append("retrieval/search systems")
    if ranking >= 0.5:
        strengths.append("ranking/recommendation systems")
    if production >= 0.5:
        strengths.append("production deployment")
    if evaluation >= 0.5:
        strengths.append("evaluation frameworks")

    if strengths:
        parts.append(f"Strong background in {' and '.join(strengths[:3])}")

    # ── Behavioral signals ───────────────────────────────────────────
    behavioral = scores.get("behavioral_score", 0)
    sig = candidate.redrob_signals

    behavioral_notes = []
    if sig.recruiter_response_rate >= 0.7:
        behavioral_notes.append(f"high recruiter response rate ({sig.recruiter_response_rate:.0%})")
    if sig.github_activity_score > 50:
        behavioral_notes.append("active GitHub presence")
    if sig.open_to_work_flag:
        behavioral_notes.append("open to work")
    if sig.interview_completion_rate >= 0.8:
        behavioral_notes.append(f"strong interview completion ({sig.interview_completion_rate:.0%})")

    if behavioral_notes:
        parts.append(". ".join(behavioral_notes[:2]).capitalize())

    # ── Compose final reasoning ──────────────────────────────────────
    reasoning = ". ".join(parts)
    if not reasoning.endswith("."):
        reasoning += "."

    # Keep it concise — 1-2 sentences
    if len(reasoning) > 300:
        reasoning = reasoning[:297] + "..."

    return reasoning
