"""
Reasoning generator — template-based, no LLM (TASK 5 — improved reasoning quality).

Generates 1-2 sentence reasoning using only actual candidate data.
No hallucinations. References real skills, experience, and signals.
Structured into:
  1. Technical signal
  2. Production signal
  3. Hiring signal
  4. Optional concern (Notice, Hype, Traps, Experience Outliers)
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
    
    # ── 1. Technical Signal ──────────────────────────────────────────
    tech_skills = get_matched_skills(candidate, jd_features)
    tech_skills = _deduplicate_skills(tech_skills)
    
    # Clean and singularize/pluralize duplicates (e.g. Embeddings and embedding)
    normalized_skills = []
    seen_roots = set()
    for s in tech_skills:
        root = s.lower().replace(" ", "").replace("-", "").strip()
        if root.endswith("s") and root[:-1] in seen_roots:
            continue
        if root + "s" in seen_roots:
            continue
        if root not in seen_roots:
            seen_roots.add(root)
            normalized_skills.append(s)
            
    tech_parts = []
    if normalized_skills:
        tools = [s for s in normalized_skills if s.lower() in {
            "milvus", "faiss", "pinecone", "weaviate", "elasticsearch", "opensearch", 
            "solr", "lucene", "qdrant", "pytorch", "tensorflow", "transformers", "langchain", "llamaindex"
        }]
        concepts = [s for s in normalized_skills if s not in tools]
        
        if tools:
            tech_parts.append(f"Demonstrated technical expertise with {', '.join(tools[:3])} ({yoe:.0f} YOE)")
            if concepts:
                tech_parts.append(f"focusing on {concepts[0].lower()}")
        else:
            tech_parts.append(f"Technical skills in {', '.join(concepts[:3])} ({yoe:.0f} YOE)")
    else:
        tech_parts.append(f"Background in machine learning engineering ({yoe:.0f} YOE)")


    tech_sentence = " ".join(tech_parts)
    if not tech_sentence.endswith("."):
        tech_sentence += "."

    # ── 2. Production Signal ─────────────────────────────────────────
    prod_score = scores.get("production_score", 0.0)
    career_desc = " ".join([c.description.lower() for c in candidate.career_history if c.description])
    has_scale = any(w in career_desc for w in ["scale", "million", "million users", "high traffic", "low latency", "qps", "production"])
    
    if prod_score >= 0.7 or has_scale:
        prod_sentence = "Proven experience deploying models to production and managing system scale."
    elif prod_score >= 0.4:
        prod_sentence = "Familiar with production ML pipelines and deployment workflows."
    else:
        prod_sentence = "Focuses primarily on model development and engineering."

    # ── 3. Hiring Signal ─────────────────────────────────────────────
    sig = candidate.redrob_signals
    hiring_parts = []
    if sig.recruiter_response_rate >= 0.7:
        hiring_parts.append("strong recruiter engagement")
    if sig.github_activity_score > 60:
        hiring_parts.append("active GitHub contribution history")
    if sig.open_to_work_flag:
        hiring_parts.append("actively seeking new opportunities")
        
    notice = sig.notice_period_days
    if notice <= 30:
        hiring_parts.append("available for immediate onboarding")
    elif notice <= 60:
        hiring_parts.append(f"notice period of {notice} days")
        
    if hiring_parts:
        hiring_sentence = f"Candidate displays {', '.join(hiring_parts)}."
    else:
        hiring_sentence = "Hiring readiness is within standard parameters."

    # ── 4. Optional Concern ──────────────────────────────────────────
    concerns = []
    if notice > 90:
        concerns.append(f"long notice period of {notice} days")
    
    hype_penalty = scores.get("llm_hype_penalty", 0.0)
    if hype_penalty > 0.05:
        concerns.append("AI experience leans heavily on recent LLM wrapper libraries")
        
    trap_prob = scores.get("trap_probability", 0.0)
    if trap_prob > 0.3:
        concerns.append("profile exhibits consistency anomalies")
        
    yoe = candidate.years_of_experience
    if yoe < 3.0:
        concerns.append("lower years of experience relative to senior requirements")
    elif yoe >= 15.0:
        concerns.append("experience level exceeds typical mid-to-senior target range")

    if concerns:
        concern_sentence = f"Note: {'; '.join(concerns)}."
    else:
        concern_sentence = ""

    # Combine sentences
    sentences = [tech_sentence, prod_sentence, hiring_sentence]
    if concern_sentence:
        sentences.append(concern_sentence)
        
    reasoning = " ".join(sentences)
    reasoning = reasoning.replace("..", ".").replace(" .", "").strip()
    return reasoning
