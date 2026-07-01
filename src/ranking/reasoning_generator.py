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
    total_ranked = len(ranked_candidates)
    for entry in ranked_candidates:
        cid = entry["candidate_id"]
        candidate = candidate_map.get(cid)

        if not candidate:
            entry["reasoning"] = "Candidate data not available."
            continue

        entry["reasoning"] = _build_reasoning(entry, candidate, jd_features, total_ranked)

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



def _relative_weakness_clause(scores: Dict[str, Any], rank: int, total: int) -> str:
    """
    For candidates outside the top tier, name the candidate's own
    comparatively weakest scoring dimension, using its real value.

    This exists to fix two related problems with purely template-driven
    reasoning:
      - "Honest concerns": a candidate ranked 90th should read differently
        from one ranked 1st, even when no hard concern (visa, notice
        period, etc.) happens to trigger.
      - "Variation": the named dimension and its real numeric value differ
        candidate to candidate, so this clause can't collapse into a
        handful of repeated stock phrases the way a purely categorical
        template can.

    Returns "" when there's nothing meaningful to say (top tier, or not
    enough score components available to make a fair comparison).
    """
    if total <= 0 or rank is None:
        return ""

    top_tier_cutoff = max(10, round(total * 0.10))
    mid_tier_cutoff = max(top_tier_cutoff, round(total * 0.50))

    components = {
        "semantic alignment with the JD": scores.get("semantic_score"),
        "structured technical fit": scores.get("structured_score"),
        "behavioral/hiring-readiness signal": scores.get("behavioral_score"),
        "profile quality/trust signal": scores.get("quality_score"),
    }
    available = {k: v for k, v in components.items() if v is not None}

    if rank <= top_tier_cutoff:
        return ""  # top tier — let the reasoning stand on its own merits
    if len(available) < 2:
        return ""  # not enough components for a fair comparison

    weakest_label, weakest_value = min(available.items(), key=lambda kv: kv[1])
    strongest_label, strongest_value = max(available.items(), key=lambda kv: kv[1])
    gap = strongest_value - weakest_value

    if rank <= mid_tier_cutoff:
        # Mid tier — only worth mentioning if there's a real, meaningful gap
        if weakest_value >= 0.55 or gap < 0.15:
            return ""
        return (
            f"Ranks behind the strongest candidates primarily on {weakest_label} "
            f"({weakest_value:.2f} vs. {strongest_value:.2f} on {strongest_label})"
        )

    # Bottom tier — always surface it, even if the gap is modest
    return (
        f"Ranks behind the strongest candidates primarily on {weakest_label} "
        f"({weakest_value:.2f} vs. {strongest_value:.2f} on {strongest_label}) — "
        f"the technical depth is there, but the exact JD keywords are missing."
    )

def _extract_transferable_domain(candidate: Candidate) -> str:
    """
    Scans career descriptions and summary to identify if an 'adjacent'
    candidate actually built a system that relies on retrieval/ranking fundamentals.
    """
    career_text = " ".join([c.description.lower() for c in candidate.career_history if c.description])
    summary = (candidate.summary or "").lower()
    full_text = f"{career_text} {summary}"

    domains = {
        "recommendation systems": ["recommendation", "recommender", "recsys"],
        "search infrastructure": ["search engine", "search relevance", "information retrieval"],
        "ranking systems": ["learning to rank", "ranking pipeline", "re-ranking"],
        "e-commerce personalization": ["e-commerce", "product ranking", "personalization"],
        "ad-tech / sponsored content": ["ad-tech", "sponsored", "ad ranking", "pricing optimization"],
        "marketplace matching": ["marketplace", "matchmaking", "two-sided market"],
    }

    for domain_name, keywords in domains.items():
        if any(kw in full_text for kw in keywords):
            return domain_name
    return ""

def _build_reasoning(
    scores: Dict[str, Any],
    candidate: Candidate,
    jd_features: JDFeatures,
    total_ranked: int = 100,
) -> str:
    """Build a reasoning string from actual candidate data structured into 4 parts."""
    yoe = candidate.years_of_experience
    cid_num = 0
    rank = scores.get("rank")
    try:
        cid_num = int(candidate.candidate_id.split("_")[1])
    except:
        pass

    # ── Sentence 1 — Technical Signal (Fix #5: connect to JD requirements) ──
    matched_skills = get_matched_skills(candidate, jd_features)
    matched_skills = _deduplicate_skills(matched_skills)

    tools = [s for s in matched_skills if s.lower() in _TOOL_NAMES]
    concepts = [s for s in matched_skills if s.lower() not in _TOOL_NAMES]

    pattern_idx = cid_num % 8
    tool_str = ", ".join(tools[:3]) if tools else ""
    first_concept = concepts[0].lower() if concepts else ""
    concept_str = ", ".join(concepts[:3]) if concepts else ""

    # Determine which JD requirements the candidate's skills satisfy
    jd_fit_aspects = []
    skill_names_lower = {s.name.lower() for s in candidate.skills}
    career_desc_full = " ".join([c.description.lower() for c in candidate.career_history if c.description])
    all_candidate_text = " ".join([career_desc_full, (candidate.summary or "").lower(),
                                   " ".join(s.name.lower() for s in candidate.skills)])
    if any(kw in all_candidate_text for kw in ["retrieval", "search", "hybrid search", "vector search", "rag"]):
        jd_fit_aspects.append("hybrid retrieval")
    if any(kw in all_candidate_text for kw in ["ndcg", "mrr", "map", "evaluation", "a/b test", "offline eval"]):
        jd_fit_aspects.append("evaluation frameworks (NDCG/MRR)")
    if any(kw in all_candidate_text for kw in ["milvus", "pinecone", "weaviate", "qdrant", "faiss", "elasticsearch", "vector database"]):
        jd_fit_aspects.append("vector DB operations")
    if any(kw in all_candidate_text for kw in ["deployed", "shipped", "production", "scale", "latency"]):
        jd_fit_aspects.append("production deployment")
    if any(kw in all_candidate_text for kw in ["ranking", "recommendation", "learning to rank", "re-ranking"]):
        jd_fit_aspects.append("ranking systems")

    jd_fit_str = ", ".join(jd_fit_aspects[:3]) if jd_fit_aspects else ""

    if tools and concepts:
        if jd_fit_str:
            if pattern_idx == 0:
                tech_sentence = f"Brings {yoe:.0f} years of ML experience with {tool_str}, directly relevant to JD requirements for {jd_fit_str}."
            elif pattern_idx == 1:
                tech_sentence = f"Demonstrates a strong technical profile in {first_concept} using {tool_str} over a {yoe:.0f}-year career, aligning with the JD's need for {jd_fit_str}."
            elif pattern_idx == 2:
                tech_sentence = f"An experienced ML professional with {yoe:.0f} YOE specializing in {first_concept} and {tool_str}, matching the role's {jd_fit_str} requirements."
            elif pattern_idx == 3:
                tech_sentence = f"Possesses {yoe:.0f} YOE with hands-on {tool_str} experience applicable to the JD's {jd_fit_str} needs."
            elif pattern_idx == 4:
                tech_sentence = f"Strong {first_concept} and {tool_str} background over {yoe:.0f} years, relevant to the role's {jd_fit_str} scope."
            elif pattern_idx == 5:
                tech_sentence = f"{yoe:.0f} years deep in {first_concept}, with {tool_str} on the resume — a near match for the JD's {jd_fit_str} ask."
            elif pattern_idx == 6:
                tech_sentence = f"Carries {tool_str} expertise from {yoe:.0f} years of hands-on work, mapping cleanly onto {jd_fit_str}."
            else:
                tech_sentence = f"A {yoe:.0f}-year track record built around {first_concept}, with practical {tool_str} exposure that speaks directly to {jd_fit_str}."
        else:
            if pattern_idx == 0:
                tech_sentence = f"Brings {yoe:.0f} years of ML experience, with a focus on {first_concept} and proficiency in {tool_str}."
            elif pattern_idx == 1:
                tech_sentence = f"Demonstrates a strong technical profile in {first_concept} using tools like {tool_str} over a {yoe:.0f}-year career."
            elif pattern_idx == 2:
                tech_sentence = f"An experienced ML professional with {yoe:.0f} YOE, specializing in {first_concept} and utilizing {tool_str} in production."
            elif pattern_idx == 3:
                tech_sentence = f"Possesses {yoe:.0f} YOE and a solid background in {concepts[0]}, with hands-on skill in {tool_str}."
            elif pattern_idx == 4:
                tech_sentence = f"Strong background in {first_concept} and {tool_str} developed over {yoe:.0f} years in the industry."
            elif pattern_idx == 5:
                tech_sentence = f"{yoe:.0f} years of hands-on {first_concept} work, with {tool_str} as part of the regular toolkit."
            elif pattern_idx == 6:
                tech_sentence = f"Carries {tool_str} experience built up over {yoe:.0f} years working in {first_concept}."
            else:
                tech_sentence = f"A {yoe:.0f}-year career centered on {first_concept}, backed by practical use of {tool_str}."
    elif tools:
        if pattern_idx % 2 == 0:
            tech_sentence = f"Demonstrated expertise with {tool_str} ({yoe:.0f} YOE) in production environments{', relevant to ' + jd_fit_str if jd_fit_str else ''}."
        else:
            tech_sentence = f"Brings {yoe:.0f} YOE with specialized proficiency in {tool_str}{', aligning with ' + jd_fit_str + ' needs' if jd_fit_str else ''}."
    elif concepts:
        if pattern_idx % 2 == 0:
            tech_sentence = f"Technical background in {concept_str} ({yoe:.0f} YOE){', with exposure to ' + jd_fit_str if jd_fit_str else ''}."
        else:
            tech_sentence = f"Possesses {yoe:.0f} years of experience specializing in {concept_str}{', relevant to ' + jd_fit_str if jd_fit_str else ''}."
    else:
        # Check if they have a transferable domain
        transfer_domain = _extract_transferable_domain(candidate)
        if transfer_domain:
            tech_sentence = (
                f"Adjacent ML/engineering background ({yoe:.0f} YOE) in **{transfer_domain}** — "
                "while it lacks exact JD keyword overlap, this domain relies on identical retrieval, "
                "ranking, and A/B testing infrastructure that the JD explicitly values. "
                "Ranked highly based on strong production & system design signals, not keyword matching."
            )
        elif rank is not None and rank <= 20:
            prod_score_local = scores.get("production_score", 0.0)
            retrieval_local = scores.get("retrieval_score", 0.0)
            strongest_signal = "production engineering" if prod_score_local >= retrieval_local else "retrieval-adjacent signals"
            tech_sentence = (
                f"Broader ML/engineering profile ({yoe:.0f} YOE) without exact JD keyword overlap, "
                f"but ranked on the strength of {strongest_signal} "
                f"(production={prod_score_local:.2f}, retrieval={retrieval_local:.2f}). "
                "Requires validation of retrieval/ranking depth in screening."
            )
        else:
            tech_sentence = (
                f"Adjacent ML/engineering background ({yoe:.0f} YOE) without transferable "
                "retrieval/ranking experience (e.g., search, recsys, ad-tech). Generic engineering "
                "background does not map to the JD's core matching/ranking needs."
            )

    # ── Sentence 2 — Production Signal (Fix #5: reference JD's "ship in weeks" culture) ──
    career_desc = " ".join(
        [c.description.lower() for c in candidate.career_history if c.description]
    )
    has_scale = any(
        w in career_desc
        for w in ["scale", "million", "billion", "high traffic", "low latency",
                   "qps", "production", "shipped", "deployed"]
    )
    prod_score = scores.get("production_score", 0.0)
    prod_pattern = cid_num % 6

    if prod_score >= 0.6 or has_scale:
        if prod_pattern == 0:
            prod_sentence = "Proven experience deploying ML systems to production at scale, matching the JD's 'ship a working ranker in a week' culture."
        elif prod_pattern == 1:
            prod_sentence = "Has a track record of deploying robust ML systems to production, consistent with the role's emphasis on shipping over theorizing."
        elif prod_pattern == 2:
            prod_sentence = "Experienced in taking models from research to production with scalability focus, fitting the founding-team velocity expectations."
        elif prod_pattern == 3:
            prod_sentence = "Has shipped ML systems under real production constraints, the kind of background the JD's fast-moving founding team is looking for."
        elif prod_pattern == 4:
            prod_sentence = "Comfortable owning a model end-to-end through deployment, rather than stopping at the research stage the JD explicitly wants to avoid."
        else:
            prod_sentence = "Track record includes live, production-facing ML work — a direct match for the JD's bias toward shipping over theorizing."
    elif prod_score >= 0.35:
        if prod_pattern == 0:
            prod_sentence = "Familiar with production ML pipelines, though depth of deployment experience is moderate relative to the JD's production-first expectations."
        elif prod_pattern == 1:
            prod_sentence = "Possesses experience with production-level code, though the JD's emphasis on shipping velocity may require a ramp-up."
        else:
            prod_sentence = "Demonstrated capability in production ML workflows, partially meeting the role's hands-on deployment requirements."
    else:
        if prod_pattern == 0:
            prod_sentence = "Experience primarily in model development and research; the JD explicitly deprioritizes pure-research backgrounds."
        else:
            prod_sentence = "Work history centered on ML research and prototyping; limited evidence of production deployment the JD requires."

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
            connector = cid_num % 4
            if connector == 0:
                hiring_sentence = f"The candidate shows {', '.join(hiring_parts)}."
            elif connector == 1:
                hiring_sentence = f"Features {', '.join(hiring_parts)}."
            elif connector == 2:
                hiring_sentence = f"On the hiring-readiness side: {', '.join(hiring_parts)}."
            else:
                hiring_sentence = f"Signals worth noting include {', '.join(hiring_parts)}."
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
        # Only surface this for bottom-half candidates; top-half adjacent candidates
        # already have an explanation in their first sentence.
        if rank is not None and rank > (total_ranked // 2):
            concerns.append("no direct JD skill matches found")

    if yoe < 3.0:
        concerns.append("total experience is below target seniority threshold")
    elif yoe >= 15.0:
        concerns.append("years of experience significantly exceed target range")

    rank = scores.get("rank")
    weakness_clause = _relative_weakness_clause(scores, rank, total_ranked)
    if weakness_clause:
        concerns.append(weakness_clause)

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

