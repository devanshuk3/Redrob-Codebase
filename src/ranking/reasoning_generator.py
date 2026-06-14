"""
Reasoning generator — template-based, no LLM (TASK 12 — improved).

Generates 1-2 sentence reasoning using only actual candidate data.
No hallucinations. References real skills, experience, and signals.

Improvements:
- Deduplicated skills
- Includes concerns (notice period, missing signals)
- Varied templates for manual-review satisfaction
- Specific, factual, non-hallucinatory
"""

from typing import Any, Dict, List, Set

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
    - Years of experience and current role
    - Deduplicated matched skills
    - Domain strengths (retrieval, ranking, production, evaluation)
    - Behavioral signals
    - Concerns (notice period, hype penalty)

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


def _deduplicate_skills(skills: List[str]) -> List[str]:
    """Deduplicate skill names (case-insensitive), preserving order."""
    seen: Set[str] = set()
    result = []
    for s in skills:
        key = s.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(s)
    return result


def _build_reasoning(
    scores: Dict[str, Any],
    candidate: Candidate,
    jd_features: JDFeatures,
) -> str:
    """Build a reasoning string from actual candidate data."""
    parts = []

    # ── Experience context ───────────────────────────────────────────
    yoe = candidate.years_of_experience
    title = candidate.current_title or "professional"
    company = candidate.current_company

    if company:
        parts.append(f"{yoe:.0f} years of experience as {title} at {company}")
    else:
        parts.append(f"{yoe:.0f} years of experience as {title}")

    # ── Key matched skills (deduplicated) ────────────────────────────
    matched = get_matched_skills(candidate, jd_features)
    matched = _deduplicate_skills(matched)
    if matched:
        top_skills = matched[:5]
        parts.append(f"with relevant skills in {', '.join(top_skills)}")

    # ── Domain strengths ─────────────────────────────────────────────
    strengths = []

    retrieval = scores.get("retrieval_score", 0)
    ranking = scores.get("ranking_score", 0)
    production = scores.get("production_score", 0)
    evaluation = scores.get("evaluation_score", 0)

    if retrieval >= 0.6:
        strengths.append("retrieval and search systems")
    elif retrieval >= 0.4:
        strengths.append("search experience")

    if ranking >= 0.6:
        strengths.append("ranking and recommendation systems")
    elif ranking >= 0.4:
        strengths.append("ranking experience")

    if production >= 0.6:
        strengths.append("production ML deployment")
    elif production >= 0.4:
        strengths.append("production engineering")

    if evaluation >= 0.6:
        strengths.append("evaluation frameworks (NDCG/MRR)")
    elif evaluation >= 0.4:
        strengths.append("evaluation metrics experience")

    if strengths:
        parts.append(f"Demonstrates strong background in {' and '.join(strengths[:3])}")

    # ── Behavioral signals ───────────────────────────────────────────
    sig = candidate.redrob_signals
    behavioral_notes = []

    if sig.recruiter_response_rate >= 0.7:
        behavioral_notes.append(f"high recruiter response rate ({sig.recruiter_response_rate:.0%})")
    if sig.github_activity_score > 50:
        behavioral_notes.append("active GitHub presence")
    if sig.open_to_work_flag:
        behavioral_notes.append("actively seeking opportunities")
    if sig.interview_completion_rate >= 0.8:
        behavioral_notes.append(f"strong interview follow-through ({sig.interview_completion_rate:.0%})")

    if behavioral_notes:
        parts.append(behavioral_notes[0].capitalize())

    # ── Concerns (when present) ──────────────────────────────────────
    concerns = []

    notice = sig.notice_period_days
    if notice > 90:
        concerns.append(f"notice period of {notice} days may delay start")
    elif notice > 60:
        concerns.append(f"notice period is {notice} days")

    hype_penalty = scores.get("llm_hype_penalty", 0)
    if hype_penalty > 0.05:
        concerns.append("AI experience skews toward recent LLM frameworks")

    if concerns:
        parts.append(f"Note: {'; '.join(concerns)}")

    # ── Compose final reasoning ──────────────────────────────────────
    reasoning = ". ".join(parts)
    if not reasoning.endswith("."):
        reasoning += "."

    # Keep it concise — 1-3 sentences
    if len(reasoning) > 350:
        reasoning = reasoning[:347] + "..."

    return reasoning
