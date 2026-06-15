"""
Reasoning generator — template-based, no LLM (TASK 5 — improved reasoning quality).

Generates 3-4 sentence reasoning using only actual candidate data.
No hallucinations. References real skills, experience, and signals.
Structured into:
  1. Technical signal (tools vs concepts)
  2. Production signal (scale + deployment)
  3. Hiring signal (recruiter, GitHub, availability)
  4. Optional concern (notice, hype, consulting, skills, experience)
"""

from datetime import date
from typing import Any, Dict, List, Set

from src.ingestion.candidate_parser import Candidate
from src.features.skill_extractor import get_matched_skills
from src.features.jd_feature_mapper import JDFeatures
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Tools recognized for sentence 1 — specific infrastructure
_TOOL_NAMES = {
    "milvus", "faiss", "pinecone", "weaviate", "elasticsearch", "opensearch",
    "qdrant", "pytorch", "tensorflow", "langchain", "lora", "qlora", "peft",
    "hugging face transformers", "transformers", "llamaindex",
}


def generate_reasoning(
    ranked_candidates: List[Dict[str, Any]],
    candidate_map: Dict[str, Candidate],
    jd_features: JDFeatures,
) -> List[Dict[str, Any]]:
    """
    Generate reasoning strings for each ranked candidate.

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
    """Build a reasoning string from actual candidate data structured into 4 parts."""
    yoe = candidate.years_of_experience

    # ── Sentence 1 — Technical Signal ──
    matched_skills = get_matched_skills(candidate, jd_features)
    matched_skills = _deduplicate_skills(matched_skills)

    tools = [s for s in matched_skills if s.lower() in _TOOL_NAMES]
    concepts = [s for s in matched_skills if s.lower() not in _TOOL_NAMES]

    if tools:
        tool_str = ", ".join(tools[:3])
        if concepts:
            tech_sentence = (
                f"Demonstrated expertise with {tool_str} ({yoe:.0f} YOE), "
                f"focusing on {concepts[0].lower()}."
            )
        else:
            tech_sentence = f"Demonstrated expertise with {tool_str} ({yoe:.0f} YOE)."
    elif concepts:
        concept_str = ", ".join(concepts[:3])
        tech_sentence = f"Technical background in {concept_str} ({yoe:.0f} YOE)."
    else:
        tech_sentence = f"Adjacent ML/engineering background ({yoe:.0f} YOE)."

    # ── Sentence 2 — Production Signal ──
    career_desc = " ".join(
        [c.description.lower() for c in candidate.career_history if c.description]
    )
    has_scale = any(
        w in career_desc
        for w in ["scale", "million", "billion", "high traffic", "low latency",
                   "qps", "production", "shipped", "deployed"]
    )
    prod_score = scores.get("production_score", 0.0)

    if prod_score >= 0.6 or has_scale:
        prod_sentence = "Proven experience deploying ML systems to production at scale."
    elif prod_score >= 0.35:
        prod_sentence = "Familiar with production ML pipelines and deployment workflows."
    else:
        prod_sentence = "Experience primarily in model development and research."

    # ── Sentence 3 — Hiring Signal ──
    sig = candidate.redrob_signals
    hiring_parts = []

    if sig.recruiter_response_rate >= 0.7:
        hiring_parts.append("strong recruiter engagement")

    if sig.github_activity_score > 60:
        hiring_parts.append("active GitHub history")

    # Days since last active
    try:
        last_active = date.fromisoformat(sig.last_active_date)
        days_since = (date(2026, 6, 14) - last_active).days
    except (ValueError, TypeError, AttributeError):
        days_since = 999

    if sig.open_to_work_flag and days_since <= 30:
        hiring_parts.append("actively looking and recently active")
    elif sig.open_to_work_flag:
        hiring_parts.append("open to work")

    notice = sig.notice_period_days
    if notice <= 30:
        hiring_parts.append("immediate joiner")
    elif notice <= 60:
        hiring_parts.append(f"{notice}-day notice")

    if hiring_parts:
        hiring_sentence = f"Candidate shows {', '.join(hiring_parts)}."
    else:
        hiring_sentence = "Hiring readiness within standard parameters."

    # ── Sentence 4 — Concerns (optional) ──
    concerns = []

    if notice > 90:
        concerns.append(f"long notice period ({notice} days)")

    hype_penalty = scores.get("llm_hype_penalty", 0.0)
    if hype_penalty > 0.05:
        concerns.append("AI experience skews toward LLM wrapper libraries")

    consulting_ratio = scores.get("consulting_ratio", 0.0)
    if consulting_ratio > 0.6:
        concerns.append("high consulting exposure")

    if not matched_skills:
        concerns.append("no direct JD skill matches found")

    if yoe < 3.0:
        concerns.append("below seniority threshold")

    if yoe >= 15.0:
        concerns.append("experience exceeds target range")

    if concerns:
        concern_sentence = f"Note: {'; '.join(concerns)}."
    else:
        concern_sentence = ""

    # ── Combine ──
    sentences = [tech_sentence, prod_sentence, hiring_sentence]
    if concern_sentence:
        sentences.append(concern_sentence)

    reasoning = " ".join(sentences)
    reasoning = reasoning.replace("..", ".").replace(" .", "").strip()
    return reasoning

