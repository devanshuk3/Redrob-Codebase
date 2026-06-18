"""
Text normalization and canonicalization utilities.
"""

import re
import unicodedata
from functools import lru_cache
from typing import List


def normalize_text(text: str) -> str:
    """Lowercase, strip, and normalize unicode."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.strip().lower()
    return text


@lru_cache(maxsize=8192)
def canonicalize_skill(skill: str) -> str:
    """
    Normalize a skill name for matching.
    'React.js' → 'reactjs', 'Machine Learning' → 'machine learning'
    """
    if not skill:
        return ""
    s = normalize_text(skill)
    # Remove dots/dashes/underscores for variant matching
    s = re.sub(r"[.\-_/]", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_keywords(text: str) -> List[str]:
    """Extract individual words from text, lowercased."""
    if not text:
        return []
    text = normalize_text(text)
    # Remove non-alphanumeric except spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    return [w for w in words if len(w) > 1]


def build_text_blob(parts: List[str], max_chars: int = 2000) -> str:
    """
    Concatenate text parts into a single blob, truncated to max_chars.
    Useful for building semantic documents.
    """
    combined = " ".join(p.strip() for p in parts if p and p.strip())
    if len(combined) > max_chars:
        combined = combined[:max_chars]
    return combined


def contains_any(text: str, keywords: List[str]) -> bool:
    """Check if text contains any of the given keywords."""
    if not text:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def count_keyword_matches(text: str, keywords: List[str]) -> int:
    """Count how many keywords appear in the text."""
    if not text:
        return 0
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)
