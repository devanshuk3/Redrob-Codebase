"""
Honeypot / synthetic profile detection (TASK 7 — strengthened with consistency checks).

Produces a quality_score (0–1) per candidate. Low scores indicate
suspicious profiles that should be filtered out.

New consistency checks:
  - Headline vs Career mismatch
  - Summary vs Career mismatch
  - Education vs Experience mismatch
  - Skill stuffing (unrelated domain diversity)
"""

from typing import Optional

from src.ingestion.candidate_parser import Candidate
from src.utils.config import Config
from src.utils.constants import UNRELATED_SKILL_DOMAINS
from src.utils.date_utils import years_since_graduation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _check_non_tech_role(candidate: Candidate) -> bool:
    """Check if the candidate's current role is non-technical."""
    title = candidate.current_title
    if not title and candidate.career_history:
        for job in candidate.career_history:
            if job.title:
                title = job.title
                break
    if not title:
        return False
        
    title_lower = title.lower()
    non_tech_words = {
        "hr", "human resources", "recruiter", "talent acquisition", "designer", 
        "graphic", "marketing", "sales", "account", "finance", "product manager", 
        "project manager", "writer", "content", "operations", "social media",
        "business development", "bde", "recruitment", "head of talent"
    }
    tech_words = {"engineer", "developer", "scientist", "programmer", "tech", "architect", "coder"}
    
    has_non_tech = any(w in title_lower for w in non_tech_words)
    has_tech = any(w in title_lower for w in tech_words)
    
    return has_non_tech and not has_tech


def _has_relevant_tech_experience(candidate: Candidate) -> bool:
    """Check if the candidate has at least one relevant technical/engineering role in their history."""
    RELEVANT_KEYWORDS = {
        "software", "developer", "programmer", "architect", "scientist", "data", 
        "machine learning", "ml", "ai", "deep learning", "nlp", "vision", "search", 
        "retrieval", "ranking", "recommender", "recommendation", "systems", "backend", 
        "fullstack", "full stack", "frontend", "devops", "infrastructure", "cloud"
    }
    EXCLUDED_KEYWORDS = {
        "mechanical", "civil", "electrical", "chemical", "industrial", "hardware",
        "qa", "test", "testing", "support", "sales", "marketing", "hr", "recruiter",
        "designer", "writer", "customer service", "customer support"
    }
    
    # Check current title
    if candidate.current_title:
        title = candidate.current_title.lower()
        has_rel = any(kw in title for kw in RELEVANT_KEYWORDS)
        has_excl = any(kw in title for kw in EXCLUDED_KEYWORDS)
        if has_rel and not has_excl:
            return True
            
    # Check career history
    for job in candidate.career_history:
        if job.title:
            title = job.title.lower()
            has_rel = any(kw in title for kw in RELEVANT_KEYWORDS)
            has_excl = any(kw in title for kw in EXCLUDED_KEYWORDS)
            if has_rel and not has_excl:
                return True
                
    return False


def _compute_consistency_score(candidate: Candidate) -> float:
    """
    Compute profile consistency score (TASK 7 — new).
    
    Checks for internal contradictions within a candidate's profile:
    1. Headline vs Career mismatch
    2. Summary vs Career mismatch
    3. Skill stuffing (unrelated domain diversity)
    
    Returns:
        Score in [0.0, 1.0] where 1.0 = fully consistent.
    """
    penalties = 0.0
    max_penalty = 3.0

    # ── 1. Headline vs Career mismatch ──────────────────────────────
    headline = (candidate.headline or "").lower()
    if headline and candidate.career_history:
        # Check if headline domain matches any career description
        headline_domains = _extract_domain_signals(headline)
        career_text = " ".join(
            (c.description or "").lower() + " " + (c.title or "").lower()
            for c in candidate.career_history
        )
        career_domains = _extract_domain_signals(career_text)

        if headline_domains and career_domains:
            overlap = headline_domains & career_domains
            if not overlap and len(headline_domains) > 0:
                penalties += 0.8

    # ── 2. Summary vs Career mismatch ───────────────────────────────
    summary = (candidate.summary or "").lower()
    if summary and candidate.career_history:
        summary_domains = _extract_domain_signals(summary)
        if not career_domains:
            career_text = " ".join(
                (c.description or "").lower() for c in candidate.career_history
            )
            career_domains = _extract_domain_signals(career_text)
        
        if summary_domains and career_domains:
            overlap = summary_domains & career_domains
            if not overlap and len(summary_domains) > 0:
                penalties += 0.7

    # ── 3. Skill stuffing (unrelated domain diversity) ──────────────
    if candidate.skills:
        skill_names = [s.name.lower() for s in candidate.skills]
        matched_unrelated_domains = set()
        for domain, keywords in UNRELATED_SKILL_DOMAINS.items():
            for skill in skill_names:
                if any(kw in skill for kw in keywords):
                    matched_unrelated_domains.add(domain)
                    break
        # If 3+ unrelated domains present in skill list → suspicious
        if len(matched_unrelated_domains) >= 3:
            penalties += 1.0
        elif len(matched_unrelated_domains) >= 2:
            penalties += 0.5

    consistency = max(0.0, 1.0 - (penalties / max_penalty))
    return round(consistency, 4)


def _extract_domain_signals(text: str) -> set:
    """Extract domain signal categories from text."""
    domains = set()
    domain_map = {
        "ml_ai": ["machine learning", "deep learning", "neural", "model training",
                   "nlp", "natural language", "embeddings", "transformers", "ai"],
        "search_retrieval": ["search", "retrieval", "ranking", "recommendation",
                            "matching", "indexing", "vector", "embedding"],
        "engineering": ["software", "engineer", "developer", "programming",
                       "backend", "frontend", "fullstack", "devops"],
        "data": ["data science", "data analysis", "analytics", "data engineer",
                "data pipeline", "etl", "warehouse"],
        "management": ["manager", "director", "head of", "vp", "chief",
                      "lead", "management"],
        "marketing": ["marketing", "sales", "business development", "advertising",
                     "seo", "content"],
        "hr": ["recruiter", "recruitment", "talent", "hr", "human resources"],
    }
    for domain, keywords in domain_map.items():
        if any(kw in text for kw in keywords):
            domains.add(domain)
    return domains


def compute_quality_score(candidate: Candidate) -> float:
    """
    Compute a quality score [0, 1] for a candidate.
    Higher = more trustworthy. Lower = more likely honeypot/synthetic.

    Checks:
    1. Non-tech role mismatch (immediate filter)
    2. Missing relevant tech experience (immediate filter)
    3. Experience timeline consistency
    4. Skill explosion
    5. Salary vs experience consistency
    6. Title progression sanity
    7. Experience-skill mismatch
    8. Career duration consistency
    9. Profile consistency (TASK 7 — new)
    """
    if _check_non_tech_role(candidate):
        return 0.0

    if candidate.years_of_experience >= 1.0 and not _has_relevant_tech_experience(candidate):
        return 0.0

    penalties = 0.0
    max_penalty = 10.0  # increased to accommodate new checks

    # ── 1. Experience Timeline Inconsistency ─────────────────────────
    # If graduation was recent but claimed experience is huge
    earliest_grad = _earliest_graduation_year(candidate)
    if earliest_grad:
        years_since_grad = years_since_graduation(earliest_grad)
        claimed_exp = candidate.years_of_experience
        # Allow some slack (part-time, internships)
        if claimed_exp > years_since_grad + 2:
            # Impossible timeline
            penalties += 1.0
        elif claimed_exp > years_since_grad + 1:
            penalties += 0.5

    # ── 2. Skill Explosion ───────────────────────────────────────────
    num_skills = len(candidate.skills)
    if num_skills > Config.MAX_SKILLS_REASONABLE:
        penalties += 1.0
    elif num_skills > 40:
        penalties += 0.5

    # Check for unrealistic skill diversity (many unrelated domains)
    if num_skills > 0:
        skill_names = [s.name.lower() for s in candidate.skills]
        unique_domains = _count_skill_domains(skill_names)
        if unique_domains > 10 and num_skills > 30:
            penalties += 0.5

    # ── 3. Salary Inconsistency ──────────────────────────────────────
    salary_range = candidate.redrob_signals.expected_salary_range_inr_lpa
    max_salary = salary_range.get("max", 0)
    if (candidate.years_of_experience < Config.LOW_EXP_THRESHOLD_YEARS
            and max_salary > Config.MAX_SALARY_FOR_LOW_EXP):
        penalties += 0.75

    # ── 4. Title Progression Problems ────────────────────────────────
    if len(candidate.career_history) >= 2:
        title_issue = _check_title_progression(candidate)
        penalties += title_issue

    # ── 5. Experience-Skill Mismatch ─────────────────────────────────
    # Very short career but extremely large skill inventory
    if candidate.years_of_experience < 1 and num_skills > 20:
        penalties += 0.75
    elif candidate.years_of_experience < 2 and num_skills > 35:
        penalties += 0.5

    # ── 6. Career Duration Consistency ───────────────────────────────
    total_career_months = sum(
        c.duration_months for c in candidate.career_history
        if c.duration_months > 0
    )
    total_career_years = total_career_months / 12.0
    claimed = candidate.years_of_experience
    if claimed > 0 and total_career_years > 0:
        ratio = total_career_years / claimed
        # If summed durations are way off from claimed experience
        if ratio > 2.0 or ratio < 0.3:
            penalties += 0.5

    # ── 7. Only Consulting Companies (JD Exclusions) ──────────────────
    if candidate.career_history:
        companies = [c.company.lower() for c in candidate.career_history if c.company]
        if companies:
            CONSULTING_COMPANIES = {"tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini"}
            only_consulting = all(
                any(cc in comp for cc in CONSULTING_COMPANIES)
                for comp in companies
            )
            if only_consulting:
                penalties += 1.0

    # ── 8. Title-Chaser / Frequent Job Hopper ─────────────────────────
    if len(candidate.career_history) >= 3:
        companies_seen = set()
        stints = []
        for c in candidate.career_history:
            if c.company and c.duration_months > 0:
                companies_seen.add(c.company.lower())
                stints.append(c.duration_months)
        if len(companies_seen) >= 3 and stints:
            avg_months = sum(stints) / len(stints)
            if avg_months < 18.0:
                penalties += 0.5

    # ── 9. Computer Vision/Speech/Robotics without NLP/IR ─────────────
    if num_skills > 0:
        skill_names = [s.name.lower() for s in candidate.skills]
        cv_keywords = {"computer vision", "opencv", "speech recognition", "robotics", "speech", "audio", "image processing", "cnn", "yolo", "object detection", "image segmentation"}
        nlp_keywords = {"nlp", "natural language processing", "llm", "rag", "embeddings", "retrieval", "search", "ranking", "recommendation", "recommender", "information retrieval"}
        
        has_cv = any(any(kw in s for kw in cv_keywords) for s in skill_names)
        has_nlp = any(any(kw in s for kw in nlp_keywords) for s in skill_names)
        
        if has_cv and not has_nlp:
            penalties += 0.75

    # ── 10. Profile Consistency (TASK 7 — new) ───────────────────────
    consistency = _compute_consistency_score(candidate)
    # Convert consistency to penalty: low consistency → high penalty
    consistency_penalty = (1.0 - consistency) * 1.5
    penalties += consistency_penalty

    # ── Compute final score ──────────────────────────────────────────
    quality = max(0.0, 1.0 - (penalties / max_penalty))
    return round(quality, 4)


def _earliest_graduation_year(candidate: Candidate) -> Optional[int]:
    """Find the earliest graduation year from education history."""
    years = []
    for edu in candidate.education:
        if edu.end_year and isinstance(edu.end_year, (int, float)):
            years.append(int(edu.end_year))
    return min(years) if years else None


def _count_skill_domains(skill_names: list) -> int:
    """Rough count of distinct skill domains."""
    domains = {
        "web": ["react", "angular", "vue", "html", "css", "javascript", "typescript", "django", "flask", "node"],
        "data": ["pandas", "numpy", "sql", "tableau", "power bi", "excel", "data analysis"],
        "ml": ["machine learning", "deep learning", "tensorflow", "pytorch", "keras", "scikit"],
        "devops": ["docker", "kubernetes", "aws", "azure", "gcp", "ci/cd", "terraform"],
        "mobile": ["android", "ios", "flutter", "react native", "swift", "kotlin"],
        "systems": ["c++", "c", "rust", "go", "systems programming", "linux"],
        "finance": ["accounting", "finance", "trading", "banking"],
        "marketing": ["seo", "marketing", "content", "advertising"],
        "design": ["figma", "photoshop", "ui/ux", "design"],
        "security": ["cybersecurity", "penetration", "security", "encryption"],
        "database": ["mongodb", "postgresql", "mysql", "redis", "cassandra"],
        "nlp": ["nlp", "llm", "transformers", "bert", "gpt", "embeddings"],
    }
    matched = set()
    for skill in skill_names:
        for domain, keywords in domains.items():
            if any(kw in skill for kw in keywords):
                matched.add(domain)
    return len(matched)


def _check_title_progression(candidate: Candidate) -> float:
    """Check for suspicious title progression patterns."""
    SENIOR_TITLES = {"cto", "ceo", "vp", "vice president", "director", "head", "chief", "principal", "staff"}
    JUNIOR_TITLES = {"intern", "trainee", "junior", "fresher", "entry level", "associate"}

    titles = [c.title.lower() for c in candidate.career_history if c.title]
    if len(titles) < 2:
        return 0.0

    penalty = 0.0

    # Check for senior → junior regression (suspicious)
    for i in range(len(titles) - 1):
        current = titles[i]
        next_title = titles[i + 1]
        is_current_senior = any(t in current for t in SENIOR_TITLES)
        is_next_junior = any(t in next_title for t in JUNIOR_TITLES)

        if is_current_senior and is_next_junior:
            penalty += 0.5

    # Check for junior → C-suite in < 2 years
    if len(candidate.career_history) >= 2:
        first = candidate.career_history[-1]  # earliest
        last = candidate.career_history[0]    # most recent
        first_junior = any(t in first.title.lower() for t in JUNIOR_TITLES)
        last_senior = any(t in last.title.lower() for t in SENIOR_TITLES)
        if first_junior and last_senior:
            total_months = sum(c.duration_months for c in candidate.career_history)
            if total_months < 24:
                penalty += 0.75

    return min(penalty, 1.0)
