"""
JD Feature Mapper (CHANGE 1 — no hardcoded JD).

Dynamically parses any job description to extract:
- must_have_skills
- preferred_skills
- negative_signals
- experience_range
- domain_keywords
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from src.utils.io_utils import read_text_file
from src.utils.text_utils import normalize_text
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class JDFeatures:
    """Structured representation of a parsed job description."""
    must_have_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    negative_signals: List[str] = field(default_factory=list)
    experience_range: List[float] = field(default_factory=list)  # [min, max]
    domain_keywords: List[str] = field(default_factory=list)
    raw_text: str = ""

    @property
    def min_experience(self) -> float:
        return self.experience_range[0] if self.experience_range else 5.0

    @property
    def max_experience(self) -> float:
        return self.experience_range[1] if len(self.experience_range) > 1 else (self.min_experience + 4.0)

    def all_skills(self) -> List[str]:
        """Return combined must-have and preferred skills."""
        return self.must_have_skills + self.preferred_skills

    def to_dict(self) -> Dict[str, Any]:
        return {
            "must_have_skills": self.must_have_skills,
            "preferred_skills": self.preferred_skills,
            "negative_signals": self.negative_signals,
            "experience_range": self.experience_range,
            "domain_keywords": self.domain_keywords,
        }


# ── Keyword groups for classification ────────────────────────────────
_MUST_HAVE_HEADERS = [
    "required", "must have", "requirements", "mandatory",
    "essential", "minimum qualifications", "key skills",
    "what you'll need", "what we're looking for",
]
_PREFERRED_HEADERS = [
    "preferred", "nice to have", "bonus", "good to have",
    "desirable", "preferred qualifications", "additional skills",
    "what would make you stand out",
]
_NEGATIVE_HEADERS = [
    "not suitable", "do not apply", "exclusions",
]

# Common skill patterns to extract from free text
_SKILL_PATTERNS = [
    r"python", r"java(?:script)?", r"c\+\+", r"go(?:lang)?", r"rust",
    r"react(?:\.?js)?", r"angular", r"vue(?:\.?js)?", r"node(?:\.?js)?",
    r"typescript", r"ruby", r"scala", r"kotlin", r"swift",
    r"machine\s*learning", r"deep\s*learning", r"nlp",
    r"natural\s*language\s*processing",
    r"pytorch", r"tensorflow", r"keras", r"scikit[\s-]?learn",
    r"transformers", r"hugging\s*face",
    r"llm[s]?", r"large\s*language\s*model",
    r"rag", r"retrieval[\s-]?augmented[\s-]?generation",
    r"embedding[s]?", r"vector\s*database[s]?",
    r"search\s*system[s]?", r"retrieval\s*system[s]?",
    r"ranking\s*system[s]?", r"recommendation\s*system[s]?",
    r"recommender\s*system[s]?", r"information\s*retrieval",
    r"fine[\s-]?tun(?:ing|e)", r"lora", r"qlora", r"peft",
    r"mlops", r"docker", r"kubernetes",
    r"aws", r"azure", r"gcp", r"google\s*cloud",
    r"sql", r"nosql", r"postgresql", r"mongodb", r"redis",
    r"elasticsearch", r"milvus", r"pinecone", r"weaviate", r"faiss",
    r"kafka", r"airflow", r"spark",
    r"langchain", r"llamaindex",
    r"ci[\s/]?cd", r"git",
    r"a/b\s*test(?:ing)?", r"ndcg", r"mrr", r"map",
    r"precision@k", r"recall@k",
    r"data\s*pipeline[s]?", r"feature\s*engineering",
    r"model\s*serving", r"model\s*deployment",
    r"distributed\s*system[s]?",
]

# Domain keywords to extract
_DOMAIN_KEYWORDS = [
    "search", "retrieval", "ranking", "recommendation", "matching",
    "personalization", "relevance", "indexing", "embedding",
    "inference", "serving", "production", "deployment", "monitoring",
    "evaluation", "metrics", "pipeline", "real-time", "batch",
    "scalable", "distributed", "microservice",
]


def parse_job_description(filepath: Optional[str] = None, jd_text: Optional[str] = None) -> JDFeatures:
    """
    Parse a job description file or text into structured features.

    Args:
        filepath: Path to job_description.md
        jd_text: Raw JD text (alternative to file)

    Returns:
        JDFeatures with extracted skills, experience range, domain keywords.
    """
    if jd_text is None:
        if filepath is None:
            raise ValueError("Either filepath or jd_text must be provided")
        jd_text = read_text_file(filepath)

    jd_lower = jd_text.lower()
    features = JDFeatures(raw_text=jd_text)

    # ── Extract skills by section ────────────────────────────────────
    sections = _split_into_sections(jd_text)

    must_have_skills = set()
    preferred_skills = set()

    for header, content in sections.items():
        header_lower = header.lower()
        content_lower = content.lower()

        is_must_have = any(h in header_lower for h in _MUST_HAVE_HEADERS)
        is_preferred = any(h in header_lower for h in _PREFERRED_HEADERS)
        is_negative = any(h in header_lower for h in _NEGATIVE_HEADERS)

        extracted = _extract_skills_from_text(content_lower)

        if is_negative:
            features.negative_signals.extend(extracted)
        elif is_must_have:
            must_have_skills.update(extracted)
        elif is_preferred:
            preferred_skills.update(extracted)
        else:
            # Default: treat as must-have if not clearly preferred
            must_have_skills.update(extracted)

    # If no sections detected, extract from full text
    if not must_have_skills:
        must_have_skills = set(_extract_skills_from_text(jd_lower))

    features.must_have_skills = sorted(must_have_skills)
    features.preferred_skills = sorted(preferred_skills - must_have_skills)

    # ── Extract experience range ─────────────────────────────────────
    features.experience_range = _extract_experience_range(jd_lower)

    # ── Extract domain keywords ──────────────────────────────────────
    found_domains = [kw for kw in _DOMAIN_KEYWORDS if kw in jd_lower]
    features.domain_keywords = sorted(set(found_domains))

    logger.info(
        f"JD parsed: {len(features.must_have_skills)} must-have skills, "
        f"{len(features.preferred_skills)} preferred, "
        f"experience range: {features.experience_range}, "
        f"{len(features.domain_keywords)} domain keywords"
    )

    return features


def _split_into_sections(text: str) -> Dict[str, str]:
    """Split markdown/text into header → content sections."""
    sections = {}
    current_header = "overview"
    current_content = []

    for line in text.split("\n"):
        stripped = line.strip()
        # Detect markdown headers
        if stripped.startswith("#") or (stripped and stripped.endswith(":") and len(stripped) < 80):
            if current_content:
                sections[current_header] = "\n".join(current_content)
            current_header = stripped.lstrip("#").strip().rstrip(":")
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections[current_header] = "\n".join(current_content)

    return sections


def _extract_skills_from_text(text: str) -> List[str]:
    """Extract skill names from text using regex patterns."""
    found = []
    for pattern in _SKILL_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            normalized = normalize_text(match)
            if normalized and normalized not in found:
                found.append(normalized)
    return found


def _extract_experience_range(text: str) -> List[float]:
    """Extract experience range (e.g., '5-9 years') from JD text."""
    patterns = [
        r"(\d+)\s*[-–to]+\s*(\d+)\s*(?:years?|yrs?)",
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience",
        r"minimum\s*(\d+)\s*(?:years?|yrs?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                return [float(groups[0]), float(groups[1])]
            elif len(groups) == 1:
                min_exp = float(groups[0])
                return [min_exp, min_exp + 4]
    return [5.0, 9.0]  # sensible default
