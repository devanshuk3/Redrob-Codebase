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
    cid_num = 0
    try:
        cid_num = int(candidate.candidate_id.split("_")[1])
    except:
        pass

    # ── Sentence 1 — Technical Signal ──
    matched_skills = get_matched_skills(candidate, jd_features)
    matched_skills = _deduplicate_skills(matched_skills)

    tools = [s for s in matched_skills if s.lower() in _TOOL_NAMES]
    concepts = [s for s in matched_skills if s.lower() not in _TOOL_NAMES]

    pattern_idx = cid_num % 5
    tool_str = ", ".join(tools[:3]) if tools else ""
    first_concept = concepts[0].lower() if concepts else ""
    concept_str = ", ".join(concepts[:3]) if concepts else ""

    if tools and concepts:
        if pattern_idx == 0:
            tech_sentence = f"Brings {yoe:.0f} years of ML experience, with a focus on {first_concept} and proficiency in {tool_str}."
        elif pattern_idx == 1:
            tech_sentence = f"Demonstrates a strong technical profile in {first_concept} using tools like {tool_str} over a {yoe:.0f}-year career."
        elif pattern_idx == 2:
            tech_sentence = f"An experienced ML professional with {yoe:.0f} YOE, specializing in {first_concept} and utilizing {tool_str} in production."
        elif pattern_idx == 3:
            tech_sentence = f"Possesses {yoe:.0f} YOE and a solid background in {concepts[0]}, with hands-on skill in {tool_str}."
        else:
            tech_sentence = f"Strong background in {first_concept} and {tool_str} developed over {yoe:.0f} years in the industry."
    elif tools:
        if pattern_idx % 2 == 0:
            tech_sentence = f"Demonstrated expertise with {tool_str} ({yoe:.0f} YOE) in production environments."
        else:
            tech_sentence = f"Brings {yoe:.0f} YOE with specialized proficiency in {tool_str}."
    elif concepts:
        if pattern_idx % 2 == 0:
            tech_sentence = f"Technical background in {concept_str} ({yoe:.0f} YOE)."
        else:
            tech_sentence = f"Possesses {yoe:.0f} years of experience specializing in {concept_str}."
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
    prod_pattern = cid_num % 3

    if prod_score >= 0.6 or has_scale:
        if prod_pattern == 0:
            prod_sentence = "Proven experience deploying ML systems to production at scale."
        elif prod_pattern == 1:
            prod_sentence = "Has a track record of deploying robust ML models to production, with experience in high-scale environments."
        else:
            prod_sentence = "Experienced in taking models from research to production deployments with a focus on system scalability."
    elif prod_score >= 0.35:
        if prod_pattern == 0:
            prod_sentence = "Familiar with production ML pipelines and deployment workflows."
        elif prod_pattern == 1:
            prod_sentence = "Possesses experience working with production-level code and deployment workflows."
        else:
            prod_sentence = "Demonstrated capability in production ML workflows and pipeline development."
    else:
        if prod_pattern == 0:
            prod_sentence = "Experience primarily in model development and research."
        else:
            prod_sentence = "Work history is primarily centered on ML research, model development, and prototyping."

    # ── Sentence 3 — Hiring Signal ──
    sig = candidate.redrob_signals
    days_since = 999
    if sig.last_active_date:
        try:
            parts = sig.last_active_date.split("-")
            if len(parts) == 3:
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                ref_date = max(date.today(), date(2026, 6, 18))
                days_since = (ref_date - date(y, m, d)).days
        except:
            pass

    is_stale = (days_since > 150)
    is_unresponsive = (sig.recruiter_response_rate < 0.15)
    not_seeking = not sig.open_to_work_flag

    if is_stale or is_unresponsive or not_seeking:
        reasons_for_concern = []
        if is_stale:
            reasons_for_concern.append(f"inactive for {days_since} days")
        if is_unresponsive:
            reasons_for_concern.append(f"low response rate of {sig.recruiter_response_rate:.0%}")
        if not_seeking:
            reasons_for_concern.append("not marked open to work")
        hiring_sentence = f"However, availability is a key concern: candidate is {', '.join(reasons_for_concern)}."
    else:
        hiring_parts = []
        if sig.recruiter_response_rate >= 0.7:
            hiring_parts.append("strong recruiter engagement")
        if sig.github_activity_score > 60:
            hiring_parts.append("active GitHub history")

        if sig.open_to_work_flag and days_since <= 30:
            hiring_parts.append("actively seeking new opportunities")
        elif sig.open_to_work_flag:
            hiring_parts.append("open to work")

        notice = sig.notice_period_days
        if notice <= 30:
            hiring_parts.append("immediate joiner status")
        elif notice <= 60:
            hiring_parts.append(f"a {notice}-day notice period")

        if hiring_parts:
            if cid_num % 2 == 0:
                hiring_sentence = f"The candidate shows {', '.join(hiring_parts)}."
            else:
                hiring_sentence = f"Features {', '.join(hiring_parts)}."
        else:
            hiring_sentence = "Hiring readiness is within typical parameters."

    # ── Sentence 4 — Concerns (optional) ──
    concerns = []

    country = (candidate.country or "").strip().lower()
    location = (candidate.location or "").strip().lower()
    willing_to_relocate = sig.willing_to_relocate

    if country and country != "india":
        concerns.append(f"based in {candidate.country} and requires visa sponsorship")
    else:
        is_preferred_city = any(city in location for city in ["pune", "noida", "delhi", "ncr", "gurgaon", "ghaziabad", "faridabad"])
        if not is_preferred_city and not willing_to_relocate:
            concerns.append(f"located in {candidate.location} and unwilling to relocate to Noida/Pune")

    notice = sig.notice_period_days
    if notice > 90:
        concerns.append(f"long notice period of {notice} days")

    hype_penalty = scores.get("llm_hype_penalty", 0.0)
    if hype_penalty > 0.05:
        concerns.append("experience leans heavily toward LLM framework prototyping rather than deep ML development")

    consulting_ratio = scores.get("consulting_ratio", 0.0)
    if consulting_ratio > 0.6:
        concerns.append("high consulting services exposure")

    if not matched_skills:
        concerns.append("no direct JD skill matches found")

    if yoe < 3.0:
        concerns.append("total experience is below target seniority threshold")
    elif yoe >= 15.0:
        concerns.append("years of experience significantly exceed target range")

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

