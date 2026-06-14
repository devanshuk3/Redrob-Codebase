"""
Honeypot / synthetic profile detection (TASK 2 — consistency, TASK 3 — trap probability).

Produces a quality_score (0–1) per candidate. Low scores indicate
suspicious profiles that should be filtered out.

Consistency checks include:
  - Title vs experience consistency
  - Claimed vs actual experience consistency
  - Impossible education timelines
  - Skill stuffing and unrelated skill domains
  - Summary vs career content consistency
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
    Compute profile consistency score (TASK 2).
    
    Checks for internal contradictions within a candidate's profile:
    1. Title consistency (e.g. Senior title with < 3 years experience)
    2. Experience consistency (claimed years vs actual summed history duration)
    3. Education timeline consistency (impossible start/end, bachelors/masters durations)
    4. Skill consistency (diversity across unrelated domains)
    5. Summary consistency (claims retrieval but no retrieval work in history)
    
    Returns:
        Score in [0.0, 1.0] where 1.0 = fully consistent.
    """
    penalties = 0.0
    
    # ── 1. Title Consistency ──────────────────────────────
    senior_keywords = {"senior", "lead", "staff", "principal", "cto", "architect", "director", "manager", "vp", "chief", "head"}
    titles = [c.title.lower() for c in candidate.career_history if c.title]
    if candidate.current_title:
        titles.append(candidate.current_title.lower())
        
    has_senior_title = any(any(skw in t for skw in senior_keywords) for t in titles)
    if has_senior_title and candidate.years_of_experience < 3.0:
        penalties += 0.3

    # ── 2. Experience Consistency ─────────────────────────
    claimed_yoe = candidate.years_of_experience
    total_months = sum(c.duration_months for c in candidate.career_history if c.duration_months > 0)
    actual_yoe = total_months / 12.0
    
    if len(candidate.career_history) > 0:
        if abs(claimed_yoe - actual_yoe) > 3.0:
            penalties += 0.25

    # ── 3. Education Consistency ──────────────────────────
    edu_timeline_error = False
    for edu in candidate.education:
        if edu.start_year and edu.end_year:
            if edu.start_year > edu.end_year:
                edu_timeline_error = True
            duration = edu.end_year - edu.start_year
            deg = edu.degree.lower()
            if any(b in deg for b in ["bachelor", "b.tech", "b.e", "b.sc", "b.s"]):
                if duration <= 1 or duration > 7:
                    edu_timeline_error = True
            elif any(m in deg for m in ["master", "m.tech", "m.e", "m.sc", "m.s"]):
                if duration < 1 or duration > 4:
                    edu_timeline_error = True
                    
    if edu_timeline_error:
        penalties += 0.25

    # ── 4. Skill Consistency ──────────────────────────────
    if candidate.skills:
        skill_names = [s.name.lower() for s in candidate.skills]
        domains_matched = set()
        domain_keywords = {
            "design": ["photoshop", "illustrator", "figma", "sketch", "indesign", "graphic design", "ui design", "ux design"],
            "sales": ["sales", "cold calling", "lead generation", "crm", "salesforce", "business development", "marketing"],
            "accounting": ["accounting", "bookkeeping", "tax", "audit", "tally", "quickbooks", "finance"],
            "technical": ["faiss", "milvus", "rag", "retrieval", "elasticsearch", "opensearch", "python", "pytorch", "tensorflow", "deep learning", "machine learning"]
        }
        for dom, kws in domain_keywords.items():
            if any(any(kw in s for kw in kws) for s in skill_names):
                domains_matched.add(dom)
        
        if len(domains_matched) >= 3:
            penalties += 0.3

    # ── 5. Summary Consistency ────────────────────────────
    summary = (candidate.summary or "").lower()
    has_retrieval_summary = any(kw in summary for kw in ["retrieval", "search", "rag", "vector search", "milvus", "faiss"])
    
    career_desc = " ".join([c.description.lower() for c in candidate.career_history if c.description] + 
                           [c.title.lower() for c in candidate.career_history if c.title])
    has_retrieval_career = any(kw in career_desc for kw in ["retrieval", "search", "rag", "vector search", "milvus", "faiss", "elasticsearch", "hybrid search"])
    
    if has_retrieval_summary and not has_retrieval_career:
        penalties += 0.25

    consistency = max(0.0, 1.0 - penalties)
    return round(consistency, 4)


def compute_trap_probability(candidate: Candidate) -> float:
    """
    Compute trap probability for candidate (TASK 3).
    Based on:
    - low consistency score
    - skill stuffing
    - impossible timelines
    - title inflation
    - career / summary contradictions
    """
    consistency = _compute_consistency_score(candidate)
    
    prob = 0.0
    
    # 1. Low consistency score
    if consistency < 0.6:
        prob += 0.4
    elif consistency < 0.8:
        prob += 0.2
        
    # 2. Skill stuffing
    num_skills = len(candidate.skills)
    if num_skills > 50:
        prob += 0.3
    elif num_skills > 40:
        prob += 0.15
        
    # 3. Impossible timelines (education)
    edu_timeline_error = False
    for edu in candidate.education:
        if edu.start_year and edu.end_year:
            if edu.start_year > edu.end_year:
                edu_timeline_error = True
            duration = edu.end_year - edu.start_year
            deg = edu.degree.lower()
            if any(b in deg for b in ["bachelor", "b.tech", "b.e", "b.sc", "b.s"]):
                if duration <= 1 or duration > 7:
                    edu_timeline_error = True
            elif any(m in deg for m in ["master", "m.tech", "m.e", "m.sc", "m.s"]):
                if duration < 1 or duration > 4:
                    edu_timeline_error = True
    if edu_timeline_error:
        prob += 0.25
        
    # 4. Title inflation
    senior_keywords = {"senior", "lead", "staff", "principal", "cto", "architect", "director", "manager", "vp", "chief", "head"}
    titles = [c.title.lower() for c in candidate.career_history if c.title]
    if candidate.current_title:
        titles.append(candidate.current_title.lower())
    has_senior_title = any(any(skw in t for skw in senior_keywords) for t in titles)
    if has_senior_title and candidate.years_of_experience < 3.0:
        prob += 0.25
        
    # 5. Career/Summary contradictions
    summary = (candidate.summary or "").lower()
    has_retrieval_summary = any(kw in summary for kw in ["retrieval", "search", "rag", "vector search", "milvus", "faiss"])
    career_desc = " ".join([c.description.lower() for c in candidate.career_history if c.description] + 
                           [c.title.lower() for c in candidate.career_history if c.title])
    has_retrieval_career = any(kw in career_desc for kw in ["retrieval", "search", "rag", "vector search", "milvus", "faiss", "elasticsearch", "hybrid search"])
    if has_retrieval_summary and not has_retrieval_career:
        prob += 0.2

    return round(min(1.0, max(0.0, prob)), 4)


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
    6. Profile consistency (using new _compute_consistency_score)
    """
    if _check_non_tech_role(candidate):
        return 0.0

    if candidate.years_of_experience >= 1.0 and not _has_relevant_tech_experience(candidate):
        return 0.0

    penalties = 0.0
    max_penalty = 4.0

    # ── 1. Experience Timeline Inconsistency ─────────────────────────
    earliest_grad = _earliest_graduation_year(candidate)
    if earliest_grad:
        years_since_grad = years_since_graduation(earliest_grad)
        claimed_exp = candidate.years_of_experience
        if claimed_exp > years_since_grad + 2:
            penalties += 1.0
        elif claimed_exp > years_since_grad + 1:
            penalties += 0.5

    # ── 2. Skill Explosion ───────────────────────────────────────────
    num_skills = len(candidate.skills)
    if num_skills > Config.MAX_SKILLS_REASONABLE:
        penalties += 1.0
    elif num_skills > 40:
        penalties += 0.5

    # ── 3. Salary Inconsistency ──────────────────────────────────────
    salary_range = candidate.redrob_signals.expected_salary_range_inr_lpa
    max_salary = salary_range.get("max", 0)
    if (candidate.years_of_experience < Config.LOW_EXP_THRESHOLD_YEARS
            and max_salary > Config.MAX_SALARY_FOR_LOW_EXP):
        penalties += 0.75

    # ── 4. Only Consulting Companies (JD Exclusions) ──────────────────
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

    safety_factor = max(0.0, 1.0 - (penalties / max_penalty))
    consistency = _compute_consistency_score(candidate)

    # Blend consistency and safety factor into quality score
    quality = 0.60 * consistency + 0.40 * safety_factor
    return round(quality, 4)


def _earliest_graduation_year(candidate: Candidate) -> Optional[int]:
    """Find the earliest graduation year from education history."""
    years = []
    for edu in candidate.education:
        if edu.end_year and isinstance(edu.end_year, (int, float)):
            years.append(int(edu.end_year))
    return min(years) if years else None
