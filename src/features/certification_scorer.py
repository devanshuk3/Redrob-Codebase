"""
Certification scoring — lightweight bonus only (CHANGE 4 — de-prioritized).

Certifications collectively contribute <= 5% of the structured score.
"""

from typing import Any, Dict, List

from src.ingestion.candidate_parser import Candidate
from src.utils.constants import RELEVANT_CERTIFICATIONS
from src.utils.text_utils import normalize_text


def score_certifications(candidate: Candidate) -> float:
    """
    Lightweight certification bonus score [0, 1].

    Detects relevant certifications and produces a small bonus.

    Returns:
        Normalized score [0, 1].
    """
    certs = candidate.certifications
    if not certs:
        return 0.0

    relevant_count = 0

    for cert in certs:
        cert_text = ""
        if isinstance(cert, dict):
            cert_text = " ".join(str(v) for v in cert.values())
        elif isinstance(cert, str):
            cert_text = cert

        cert_lower = normalize_text(cert_text)
        if not cert_lower:
            continue

        for keyword in RELEVANT_CERTIFICATIONS:
            if keyword in cert_lower:
                relevant_count += 1
                break

    if relevant_count == 0:
        return 0.0

    # Diminishing returns: 1 cert = 0.5, 2 = 0.75, 3+ = 1.0
    score = min(1.0, 0.5 + (relevant_count - 1) * 0.25)
    return round(score, 4)


def get_relevant_certifications(candidate: Candidate) -> List[str]:
    """Return names of relevant certifications (for reasoning)."""
    result = []
    for cert in candidate.certifications:
        cert_text = ""
        if isinstance(cert, dict):
            cert_text = cert.get("name", cert.get("title", str(cert)))
        elif isinstance(cert, str):
            cert_text = cert

        cert_lower = normalize_text(cert_text)
        for keyword in RELEVANT_CERTIFICATIONS:
            if keyword in cert_lower:
                result.append(cert_text)
                break
    return result
