"""
Career analyzer — searches career descriptions for domain-relevant keywords.
Produces a general career relevance score.
"""

from src.ingestion.candidate_parser import Candidate
from src.features.jd_feature_mapper import JDFeatures
from src.utils.text_utils import count_keyword_matches


def score_career(candidate: Candidate, jd_features: JDFeatures) -> tuple:
    """
    Analyze career descriptions for domain-relevant keywords from the JD.

    Scans all career descriptions for JD domain keywords and must-have skills.

    Returns:
        Tuple of (normalized_score [0, 1], detail_dict).
        detail_dict contains: consulting_ratio, title_score, yoe_score,
        relevant_ratio, short_stints.
    """
    detail = {
        "consulting_ratio": 0.0,
        "title_score": 0.0,
        "yoe_score": 0.0,
        "relevant_ratio": 0.0,
        "short_stints": 0,
    }

    if not candidate.career_history:
        return 0.0, detail

    # Combine all career descriptions into one text
    all_descriptions = " ".join(
        c.description.lower() for c in candidate.career_history
        if c.description
    )
    all_titles = " ".join(
        c.title.lower() for c in candidate.career_history
        if c.title
    )
    combined = all_descriptions + " " + all_titles

    # Compute consulting ratio
    CONSULTING_COMPANIES = {"tcs", "tata consultancy", "infosys", "wipro",
                            "accenture", "cognizant", "capgemini", "hcl",
                            "tech mahindra", "mindtree", "mphasis", "ltimindtree"}
    companies = [c.company.lower() for c in candidate.career_history if c.company]
    if companies:
        consulting_count = sum(
            1 for comp in companies
            if any(cc in comp for cc in CONSULTING_COMPANIES)
        )
        detail["consulting_ratio"] = round(consulting_count / len(companies), 4)

    # Compute title score — how relevant are the job titles
    RELEVANT_TITLE_KW = {"engineer", "developer", "scientist", "architect",
                         "ml", "machine learning", "data", "search", "retrieval",
                         "ranking", "recommendation", "ai", "nlp"}
    if all_titles.strip():
        title_hits = sum(1 for kw in RELEVANT_TITLE_KW if kw in all_titles)
        detail["title_score"] = round(min(1.0, title_hits / 4.0), 4)

    # Compute YOE score — how well does experience match
    yoe = candidate.years_of_experience
    ideal_min = jd_features.min_experience if hasattr(jd_features, 'min_experience') else 5.0
    ideal_max = jd_features.max_experience if hasattr(jd_features, 'max_experience') else 9.0
    if ideal_min <= yoe <= ideal_max:
        detail["yoe_score"] = 1.0
    else:
        dist = min(abs(yoe - ideal_min), abs(yoe - ideal_max))
        detail["yoe_score"] = round(max(0.0, 1.0 - dist * 0.1), 4)

    # Compute relevant ratio — fraction of career entries with relevant descriptions
    relevant_count = 0
    for career in candidate.career_history:
        desc = (career.description or "").lower()
        title = (career.title or "").lower()
        text = desc + " " + title
        if any(kw in text for kw in ["ml", "machine learning", "data", "search",
                                      "retrieval", "ranking", "recommendation",
                                      "engineer", "developer", "software"]):
            relevant_count += 1
    detail["relevant_ratio"] = round(
        relevant_count / len(candidate.career_history), 4
    ) if candidate.career_history else 0.0

    # Count short stints (< 12 months)
    detail["short_stints"] = sum(
        1 for c in candidate.career_history
        if (c.duration_months or 0) > 0 and c.duration_months < 12
    )

    if not combined.strip():
        return 0.0, detail

    # Use JD domain keywords + must-have skills as search terms
    keywords = jd_features.domain_keywords + jd_features.must_have_skills
    if not keywords:
        return 0.0, detail

    matches = count_keyword_matches(combined, keywords)
    # Normalize: hitting ~40% of keywords is a strong signal
    score = min(1.0, matches / (len(keywords) * 0.4))

    return round(score, 4), detail


def get_career_highlights(candidate: Candidate, jd_features: JDFeatures) -> list:
    """Return career keywords found (for reasoning)."""
    highlights = []
    keywords = jd_features.domain_keywords + jd_features.must_have_skills[:10]
    for career in candidate.career_history:
        desc = (career.description or "").lower()
        title = (career.title or "").lower()
        text = desc + " " + title
        for kw in keywords:
            if kw in text and kw not in highlights:
                highlights.append(kw)
    return highlights[:10]
