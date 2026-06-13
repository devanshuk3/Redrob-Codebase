"""
Retrieval system experience scorer (CHANGE 2).

Detects evidence of search/retrieval/RAG/vector-search experience
in career descriptions, titles, and skills.
"""

from src.ingestion.candidate_parser import Candidate
from src.utils.constants import RETRIEVAL_KEYWORDS
from src.utils.text_utils import count_keyword_matches, canonicalize_skill


def score_retrieval(candidate: Candidate) -> float:
    """
    Score retrieval/search system experience.

    Scans career descriptions, titles, skills, headline, and summary
    for retrieval-domain keywords.

    Returns:
        Normalized score [0, 1].
    """
    # Build combined text from all relevant fields
    parts = []

    # Career descriptions and titles
    for career in candidate.career_history:
        if career.description:
            parts.append(career.description.lower())
        if career.title:
            parts.append(career.title.lower())

    # Headline and summary
    if candidate.headline:
        parts.append(candidate.headline.lower())
    if candidate.summary:
        parts.append(candidate.summary.lower())

    # Skill names
    skill_text = " ".join(s.name.lower() for s in candidate.skills if s.name)
    parts.append(skill_text)

    combined = " ".join(parts)

    if not combined.strip():
        return 0.0

    matches = count_keyword_matches(combined, RETRIEVAL_KEYWORDS)

    # Normalize: hitting 5+ retrieval keywords is very strong
    if matches >= 8:
        score = 1.0
    elif matches >= 5:
        score = 0.85
    elif matches >= 3:
        score = 0.65
    elif matches >= 2:
        score = 0.45
    elif matches >= 1:
        score = 0.25
    else:
        score = 0.0

    # Bonus for retrieval-related skill names
    retrieval_skill_keywords = {"retrieval", "search", "rag", "vector", "elasticsearch",
                                "milvus", "pinecone", "weaviate", "faiss", "solr", "lucene"}
    skill_bonus = 0.0
    for s in candidate.skills:
        canon = canonicalize_skill(s.name)
        if any(kw in canon for kw in retrieval_skill_keywords):
            skill_bonus += 0.1

    score = min(1.0, score + min(0.2, skill_bonus))
    return round(score, 4)
