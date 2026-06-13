"""
Text builder — constructs optimized semantic documents for embedding.
"""

from typing import List

from src.ingestion.candidate_parser import Candidate
from src.utils.text_utils import build_text_blob


def build_candidate_text(candidate: Candidate, max_chars: int = 1500) -> str:
    """
    Build a semantic text document from a candidate's profile.

    Combines headline, summary, skills, career descriptions, and education
    into a single text blob optimized for semantic embedding.

    Args:
        candidate: Parsed Candidate object.
        max_chars: Maximum characters for the output text.

    Returns:
        Concatenated text string.
    """
    parts = []

    # Headline — most concise summary of the candidate
    if candidate.headline:
        parts.append(candidate.headline)

    # Current title and company
    if candidate.current_title:
        title_str = candidate.current_title
        if candidate.current_company:
            title_str += f" at {candidate.current_company}"
        parts.append(title_str)

    # Summary — rich narrative
    if candidate.summary:
        parts.append(candidate.summary)

    # Skills — comma-separated list
    skill_names = [s.name for s in candidate.skills if s.name]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))

    # Career descriptions — most recent first
    for career in candidate.career_history[:3]:  # top 3 roles
        desc_parts = []
        if career.title:
            desc_parts.append(career.title)
        if career.company:
            desc_parts.append(f"at {career.company}")
        if career.description:
            desc_parts.append(career.description)
        if desc_parts:
            parts.append(" ".join(desc_parts))

    # Education fields
    for edu in candidate.education[:2]:
        edu_parts = []
        if edu.degree:
            edu_parts.append(edu.degree)
        if edu.field_of_study:
            edu_parts.append(f"in {edu.field_of_study}")
        if edu.institution:
            edu_parts.append(f"from {edu.institution}")
        if edu_parts:
            parts.append(" ".join(edu_parts))

    # Certifications
    for cert in candidate.certifications[:3]:
        if isinstance(cert, dict):
            cert_name = cert.get("name", cert.get("title", ""))
            if cert_name:
                parts.append(f"Certified: {cert_name}")
        elif isinstance(cert, str) and cert:
            parts.append(f"Certified: {cert}")

    return build_text_blob(parts, max_chars=max_chars)


def build_jd_text(jd_raw: str, max_chars: int = 1500) -> str:
    """
    Build a semantic document from job description text.

    Args:
        jd_raw: Raw job description markdown/text.
        max_chars: Maximum characters.

    Returns:
        Cleaned JD text for embedding.
    """
    # Strip markdown formatting but keep content
    lines = []
    for line in jd_raw.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        if line.startswith("*") or line.startswith("-"):
            line = line.lstrip("*- ").strip()
        if line:
            lines.append(line)

    return build_text_blob(lines, max_chars=max_chars)


def build_candidate_texts(candidates: List[Candidate], max_chars: int = 1500) -> List[str]:
    """Build semantic texts for a list of candidates."""
    return [build_candidate_text(c, max_chars=max_chars) for c in candidates]
