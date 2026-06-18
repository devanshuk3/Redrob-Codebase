"""
Honeypot / synthetic profile detection (TASK 2 — consistency, TASK 3 — trap probability).
Refactored to support Dynamic Calibration, chronological validation (TASK 2), false positive reduction (TASK 3),
and issue registry to prevent double penalization (TASK 4).
"""

from typing import Optional, Dict, Any, List

from src.ingestion.candidate_parser import Candidate
from src.utils.config import Config
from src.utils.constants import UNRELATED_SKILL_DOMAINS
from src.utils.date_utils import years_since_graduation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def is_heavy_tech_overlap(name1: str, name2: str) -> bool:
    """Check if both degrees are heavy technology/engineering degrees."""
    n1 = name1.lower()
    n2 = name2.lower()
    tech_keywords = ["tech", "eng", "b.e", "b.tech", "computer", "software", "information technology"]
    is_n1_tech = any(k in n1 for k in tech_keywords)
    is_n2_tech = any(k in n2 for k in tech_keywords)
    return is_n1_tech and is_n2_tech


def check_education_timeline(candidate: Candidate) -> tuple:
    """
    Validates chronological sequence: Bachelor's -> Master's -> PhD.
    Also detects suspicious overlap of multiple Bachelor's degrees.
    
    Returns (education_consistency_score, issues_list)
    """
    bachelors = []
    masters = []
    phds = []
    edu_issues = []
    penalties = 0.0

    for edu in candidate.education:
        if not edu.start_year or not edu.end_year:
            continue
        try:
            start = int(edu.start_year)
            end = int(edu.end_year)
        except (ValueError, TypeError):
            continue

        if start > end:
            edu_issues.append(f"Start year {start} > End year {end} for {edu.degree}")
            penalties += 0.5
            continue

        duration = end - start
        deg_lower = (edu.degree or "").lower()

        # Duration validation
        if any(w in deg_lower for w in ["bachelor", "b.tech", "b.e", "b.sc", "b.s", "b.a", "bba", "bca", "b.com"]):
            bachelors.append((start, end, edu.degree))
            if duration <= 1 or duration > 7:
                edu_issues.append(f"Suspicious Bachelor's duration ({duration} years): {edu.degree}")
                penalties += 0.2
        elif any(w in deg_lower for w in ["master", "m.tech", "m.e", "m.sc", "m.s", "mba", "mca", "pgdm"]):
            masters.append((start, end, edu.degree))
            if duration < 1 or duration > 4:
                edu_issues.append(f"Suspicious Master's duration ({duration} years): {edu.degree}")
                penalties += 0.2
        elif any(w in deg_lower for w in ["phd", "ph.d", "doctor", "doctorate"]):
            phds.append((start, end, edu.degree))
            if duration < 2 or duration > 8:
                edu_issues.append(f"Suspicious PhD duration ({duration} years): {edu.degree}")
                penalties += 0.1

    # 1. Chronological validation across levels (Bachelors -> Masters -> PhD)
    for b_start, b_end, b_name in bachelors:
        for m_start, m_end, m_name in masters:
            if m_end < b_end or m_start < b_start:
                edu_issues.append(f"Chronology error: Master's {m_name} ({m_start}-{m_end}) starts/ends before Bachelor's {b_name} ({b_start}-{b_end})")
                penalties += 0.4

        for p_start, p_end, p_name in phds:
            if p_end < b_end or p_start < b_start:
                edu_issues.append(f"Chronology error: PhD {p_name} ({p_start}-{p_end}) starts/ends before Bachelor's {b_name} ({b_start}-{b_end})")
                penalties += 0.4

    for m_start, m_end, m_name in masters:
        for p_start, p_end, p_name in phds:
            if p_end < m_end or p_start < m_start:
                edu_issues.append(f"Chronology error: PhD {p_name} ({p_start}-{p_end}) starts/ends before Master's {m_name} ({m_start}-{m_end})")
                penalties += 0.4

    # 2. Overlap validation for multiple Bachelor's degrees
    if len(bachelors) > 1:
        for i in range(len(bachelors)):
            for j in range(i + 1, len(bachelors)):
                start1, end1, name1 = bachelors[i]
                start2, end2, name2 = bachelors[j]
                overlap = max(0, min(end1, end2) - max(start1, start2))
                if overlap > 1.0 and is_heavy_tech_overlap(name1, name2):
                    edu_issues.append(f"Suspicious overlap of Bachelor's degrees: {name1} and {name2} overlap by {overlap} years")
                    penalties += 0.3

    score = max(0.0, 1.0 - penalties)
    return score, edu_issues


def _is_domain_supported_by_career(domain: str, candidate: Candidate) -> bool:
    """Check if a matched skill domain is supported by the candidate's actual roles/summaries."""
    career_text = " ".join([
        (j.title or "").lower() + " " + (j.description or "").lower() + " " + (j.company or "").lower()
        for j in candidate.career_history
    ] + [(candidate.current_title or "").lower(), (candidate.summary or "").lower()])

    # Founders/Executives have broad multi-disciplinary responsibilities
    is_founder = any(w in career_text for w in ["founder", "co-founder", "ceo", "chief executive", "president", "owner", "partner"])
    if is_founder:
        return True

    if domain == "design":
        design_kws = ["design", "ux", "ui", "graphic", "creative", "artist", "illustrator", "figma"]
        return any(w in career_text for w in design_kws)
    elif domain == "sales":
        sales_kws = ["sales", "cold call", "lead gen", "crm", "salesforce", "business development", "marketing", "growth", "account manager"]
        return any(w in career_text for w in sales_kws)
    elif domain == "accounting":
        acct_kws = ["accounting", "bookkeep", "tax", "audit", "tally", "quickbooks", "finance", "financial", "treasurer"]
        return any(w in career_text for w in acct_kws)
    elif domain == "technical":
        tech_kws = ["engineer", "developer", "scientist", "architect", "programmer", "tech", "code", "cto", "software", "data"]
        return any(w in career_text for w in tech_kws)

    return False


def analyze_candidate_issues(candidate: Candidate) -> Dict[str, Any]:
    """
    Runs all safety, timeline, and consistency checks exactly once for the candidate.
    Registers discovered issues to prevent double-penalization.
    """
    issues = {}

    # 1. Title Inconsistency
    senior_keywords = {"senior", "lead", "staff", "principal", "cto", "architect", "director", "manager", "vp", "chief", "head"}
    titles = [c.title.lower() for c in candidate.career_history if c.title]
    if candidate.current_title:
        titles.append(candidate.current_title.lower())
    has_senior_title = any(any(skw in t for skw in senior_keywords) for t in titles)
    if has_senior_title and candidate.years_of_experience < 3.0:
        issues["TITLE_INCONSISTENCY"] = {"penalty": 0.3}

    # 2. Experience Discrepancy
    claimed_yoe = candidate.years_of_experience
    total_months = sum(c.duration_months for c in candidate.career_history if c.duration_months > 0)
    actual_yoe = total_months / 12.0
    if len(candidate.career_history) > 0 and abs(claimed_yoe - actual_yoe) > 3.0:
        issues["EXPERIENCE_DISCREPANCY"] = {"penalty": 0.25}

    # 3. Education Chronology
    edu_score, edu_issues = check_education_timeline(candidate)
    if edu_score < 1.0:
        issues["EDUCATION_CHRONOLOGY"] = {"score": edu_score, "details": edu_issues}

    # 4. Skill Domain Mismatch
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

        # Only penalize if domains are matched in skills but career history does not support them (TASK 3)
        unsupported_domains = []
        for dom in domains_matched:
            if not _is_domain_supported_by_career(dom, candidate):
                unsupported_domains.append(dom)

        if len(unsupported_domains) >= 2:
            issues["SKILL_DOMAIN_MISMATCH"] = {"penalty": 0.3, "unsupported": unsupported_domains}

    # 5. Summary Contradiction
    summary = (candidate.summary or "").lower()
    has_retrieval_summary = any(kw in summary for kw in ["retrieval", "search", "rag", "vector search", "milvus", "faiss"])
    career_desc = " ".join([c.description.lower() for c in candidate.career_history if c.description] + 
                           [c.title.lower() for c in candidate.career_history if c.title])
    has_retrieval_career = any(kw in career_desc for kw in ["retrieval", "search", "rag", "vector search", "milvus", "faiss", "elasticsearch", "hybrid search"])
    if has_retrieval_summary and not has_retrieval_career:
        issues["SUMMARY_CONTRADICTION"] = {"penalty": 0.25}

    # 6. Skill Explosion
    num_skills = len(candidate.skills)
    if num_skills > Config.MAX_SKILLS_REASONABLE:
        issues["SKILL_EXPLOSION"] = {"severity": 1.0}
    elif num_skills > 40:
        issues["SKILL_EXPLOSION"] = {"severity": 0.5}

    # 7. Graduation vs Claimed Experience
    earliest_grad = _earliest_graduation_year(candidate)
    if earliest_grad:
        years_since_grad = years_since_graduation(earliest_grad)
        claimed_exp = candidate.years_of_experience
        if claimed_exp > years_since_grad + 2:
            issues["GRAD_VS_CLAIMED_EXP"] = {"severity": 1.0}
        elif claimed_exp > years_since_grad + 1:
            issues["GRAD_VS_CLAIMED_EXP"] = {"severity": 0.5}

    # 8. Salary vs Experience Mismatch
    salary_range = candidate.redrob_signals.expected_salary_range_inr_lpa
    max_salary = salary_range.get("max", 0)
    if (candidate.years_of_experience < Config.LOW_EXP_THRESHOLD_YEARS
            and max_salary > Config.MAX_SALARY_FOR_LOW_EXP):
        issues["SALARY_EXP_MISMATCH"] = {"penalty": 0.75}

    # 9. Consulting Only
    if candidate.career_history:
        companies = [c.company.lower() for c in candidate.career_history if c.company]
        if companies:
            CONSULTING_COMPANIES = {"tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini"}
            only_consulting = all(
                any(cc in comp for cc in CONSULTING_COMPANIES)
                for comp in companies
            )
            if only_consulting:
                issues["CONSULTING_ONLY"] = {"penalty": 1.0}

    return issues


def get_candidate_issues(candidate: Candidate) -> Dict[str, Any]:
    """Retrieves or computes candidate issue registry."""
    if not hasattr(candidate, "_issue_registry"):
        candidate._issue_registry = analyze_candidate_issues(candidate)
    return candidate._issue_registry


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


def _check_non_tech_role_raw(record: dict) -> bool:
    """
    Raw-dict version of _check_non_tech_role.
    Replicates the exact same logic against raw dict fields,
    producing identical pass/fail decisions.
    """
    profile = record.get("profile", {})
    title = profile.get("current_title", "") or ""
    title = str(title)

    if not title:
        career_history = record.get("career_history", [])
        if isinstance(career_history, list):
            for job in career_history:
                if isinstance(job, dict):
                    t = job.get("title", "") or ""
                    t = str(t)
                    if t:
                        title = t
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


def fast_prefilter_raw(record: dict, min_yoe: float = 3.0) -> bool:
    """
    Raw-dict version of fast_prefilter.
    Returns True if the record passes (should be kept).
    Operates on raw dict to avoid full parse_candidate cost.
    """
    if _check_non_tech_role_raw(record):
        return False
    profile = record.get("profile", {})
    try:
        yoe = float(profile.get("years_of_experience", 0))
    except (ValueError, TypeError):
        yoe = 0.0
    if yoe < 0:
        yoe = 0.0
    if yoe < min_yoe:
        return False
    return True


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
    """Compute profile consistency score using the issue registry."""
    issues = get_candidate_issues(candidate)
    penalties = 0.0
    
    if "TITLE_INCONSISTENCY" in issues:
        penalties += 0.3
    if "EXPERIENCE_DISCREPANCY" in issues:
        penalties += 0.25
    if "EDUCATION_CHRONOLOGY" in issues:
        edu_score = issues["EDUCATION_CHRONOLOGY"].get("score", 1.0)
        penalties += (1.0 - edu_score) * 0.25
    if "SKILL_DOMAIN_MISMATCH" in issues:
        penalties += 0.3
    if "SUMMARY_CONTRADICTION" in issues:
        penalties += 0.25

    return round(max(0.0, 1.0 - penalties), 4)


def compute_trap_probability(candidate: Candidate) -> float:
    """
    Compute trap probability based strictly on consistency and non-overlapping safety signals.
    Avoids double penalizing specific chronological, title, or summary contradictions.
    """
    consistency = _compute_consistency_score(candidate)
    issues = get_candidate_issues(candidate)
    prob = 0.0

    # 1. Low consistency score
    if consistency < 0.6:
        prob += 0.4
    elif consistency < 0.8:
        prob += 0.2

    # 2. Skill stuffing
    if "SKILL_EXPLOSION" in issues:
        severity = issues["SKILL_EXPLOSION"].get("severity", 0.5)
        prob += 0.3 * severity

    # 3. Salary mismatch
    if "SALARY_EXP_MISMATCH" in issues:
        prob += 0.2

    return round(min(1.0, max(0.0, prob)), 4)


def is_honeypot_candidate(candidate: Candidate) -> bool:
    """
    Check if a candidate has physically/logically impossible attributes.
    Returns True if the profile is identified as a honeypot/trap candidate.
    """
    # 1. Parse education into Bachelor's, Master's, PhD
    bachelors = []
    masters = []
    phds = []
    all_degrees = []
    
    for edu in candidate.education:
        if not edu.start_year or not edu.end_year:
            continue
        try:
            start = int(edu.start_year)
            end = int(edu.end_year)
        except (ValueError, TypeError):
            continue
        
        deg_lower = (edu.degree or "").lower()
        inst_lower = (edu.institution or "").lower()
        
        level = None
        if any(w in deg_lower for w in ["bachelor", "b.tech", "b.e", "b.sc", "b.s", "b.a", "bba", "bca", "b.com"]):
            level = "bachelor"
            bachelors.append((start, end, edu.degree, inst_lower))
        elif any(w in deg_lower for w in ["master", "m.tech", "m.e", "m.sc", "m.s", "mba", "mca", "pgdm"]):
            level = "master"
            masters.append((start, end, edu.degree, inst_lower))
        elif any(w in deg_lower for w in ["phd", "ph.d", "doctor", "doctorate"]):
            level = "phd"
            phds.append((start, end, edu.degree, inst_lower))
            
        all_degrees.append((start, end, edu.degree, inst_lower, level))
        
    # Check end-year chronology errors (undeniable logical error: higher degree completed before lower degree)
    for b_start, b_end, b_name, b_inst in bachelors:
        for m_start, m_end, m_name, m_inst in masters:
            if m_end < b_end - 1:
                return True
        for p_start, p_end, p_name, p_inst in phds:
            if p_end < b_end - 1:
                return True
    for m_start, m_end, m_name, m_inst in masters:
        for p_start, p_end, p_name, p_inst in phds:
            if p_end < m_end - 1:
                return True
                
    # Check start-year chronology errors (undeniable logical error: higher degree starts before lower degree)
    for b_start, b_end, b_name, b_inst in bachelors:
        for m_start, m_end, m_name, m_inst in masters:
            if m_start < b_start - 1 and b_inst != m_inst:
                return True
        for p_start, p_end, p_name, p_inst in phds:
            if p_start < b_start - 1 and b_inst != p_inst:
                return True
    for m_start, m_end, m_name, m_inst in masters:
        for p_start, p_end, p_name, p_inst in phds:
            if p_start < m_start - 1 and m_inst != p_inst:
                return True

    # Check overlapping Bachelor's degrees from different institutions (or same with >1 year overlap)
    if len(bachelors) > 1:
        for i in range(len(bachelors)):
            for j in range(i + 1, len(bachelors)):
                s1, e1, n1, inst1 = bachelors[i]
                s2, e2, n2, inst2 = bachelors[j]
                overlap = max(0, min(e1, e2) - max(s1, s2))
                if overlap > 1.0:
                    return True
                    
    # Check duplicate degrees at same institution
    if len(all_degrees) > 1:
        from rapidfuzz import fuzz
        for i in range(len(all_degrees)):
            for j in range(i + 1, len(all_degrees)):
                s1, e1, n1, inst1, l1 = all_degrees[i]
                s2, e2, n2, inst2, l2 = all_degrees[j]
                if inst1 and inst2 and inst1 == inst2:
                    ratio = fuzz.ratio(n1.lower(), n2.lower())
                    if ratio > 85:
                        return True
                        
    # Check skills duration anomaly (Expert/Advanced with 0 months)
    expert_zero_count = 0
    for skill in candidate.skills:
        dur = getattr(skill, "duration_months", None)
        prof = getattr(skill, "proficiency", None)
        if prof in ["expert", "advanced"] and dur == 0:
            expert_zero_count += 1
    if expert_zero_count >= 5:
        return True
        
    return False


def compute_quality_score(candidate: Candidate) -> float:
    """Compute quality score per candidate using unified issue registry."""
    if _check_non_tech_role(candidate):
        return 0.0

    if candidate.years_of_experience >= 1.0 and not _has_relevant_tech_experience(candidate):
        return 0.0

    if is_honeypot_candidate(candidate):
        return 0.0

    issues = get_candidate_issues(candidate)
    penalties = 0.0
    max_penalty = 4.0

    if "GRAD_VS_CLAIMED_EXP" in issues:
        penalties += issues["GRAD_VS_CLAIMED_EXP"].get("severity", 0.5) * 1.0
    if "SKILL_EXPLOSION" in issues:
        penalties += issues["SKILL_EXPLOSION"].get("severity", 0.5) * 1.0
    if "SALARY_EXP_MISMATCH" in issues:
        penalties += 0.75
    if "CONSULTING_ONLY" in issues:
        penalties += 1.0

    safety_factor = max(0.0, 1.0 - (penalties / max_penalty))
    consistency = _compute_consistency_score(candidate)

    quality = 0.60 * consistency + 0.40 * safety_factor
    return round(quality, 4)


def _earliest_graduation_year(candidate: Candidate) -> Optional[int]:
    """Find the earliest graduation year from education history."""
    years = []
    for edu in candidate.education:
        if edu.end_year:
            try:
                years.append(int(str(edu.end_year).strip()))
            except (ValueError, TypeError):
                continue
    return min(years) if years else None
