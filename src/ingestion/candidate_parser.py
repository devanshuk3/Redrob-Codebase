"""
Candidate parser — converts raw JSON dicts into structured Candidate dataclasses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.text_utils import normalize_text
from src.utils.date_utils import parse_date


@dataclass
class CareerEntry:
    company: str = ""
    title: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: int = 0
    is_current: bool = False
    industry: str = ""
    company_size: str = ""
    description: str = ""


@dataclass
class EducationEntry:
    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    grade: str = ""
    tier: str = ""


@dataclass
class SkillEntry:
    name: str = ""
    proficiency: str = "beginner"
    endorsements: int = 0
    duration_months: int = 0


@dataclass
class RedrobSignals:
    profile_completeness_score: float = 0.0
    signup_date: Optional[str] = None
    last_active_date: Optional[str] = None
    open_to_work_flag: bool = False
    profile_views_received_30d: int = 0
    applications_submitted_30d: int = 0
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 0.0
    skill_assessment_scores: Dict[str, Any] = field(default_factory=dict)
    connection_count: int = 0
    endorsements_received: int = 0
    notice_period_days: int = 90
    expected_salary_range_inr_lpa: Dict[str, float] = field(default_factory=dict)
    preferred_work_mode: str = ""
    willing_to_relocate: bool = False
    github_activity_score: float = -1.0
    search_appearance_30d: int = 0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False


@dataclass
class Candidate:
    candidate_id: str = ""
    # Profile
    anonymized_name: str = ""
    headline: str = ""
    summary: str = ""
    location: str = ""
    country: str = ""
    years_of_experience: float = 0.0
    current_title: str = ""
    current_company: str = ""
    current_company_size: str = ""
    current_industry: str = ""
    # Nested
    career_history: List[CareerEntry] = field(default_factory=list)
    education: List[EducationEntry] = field(default_factory=list)
    skills: List[SkillEntry] = field(default_factory=list)
    certifications: List[Dict[str, Any]] = field(default_factory=list)
    languages: List[Dict[str, str]] = field(default_factory=list)
    redrob_signals: RedrobSignals = field(default_factory=RedrobSignals)
    # Computed scores (filled during pipeline)
    quality_score: float = 1.0


def _safe_float(val, default=0.0) -> float:
    try:
        v = float(val)
        return v if v >= 0 else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0) -> int:
    try:
        v = int(val)
        return v if v >= 0 else default
    except (ValueError, TypeError):
        return default


def parse_candidate(raw: Dict[str, Any]) -> Candidate:
    """Convert a raw JSON dict into a Candidate dataclass."""
    profile = raw.get("profile", {})

    # Career history
    career = []
    for entry in raw.get("career_history", []):
        if not isinstance(entry, dict):
            continue
        career.append(CareerEntry(
            company=str(entry.get("company", "")),
            title=str(entry.get("title", "")),
            start_date=entry.get("start_date"),
            end_date=entry.get("end_date"),
            duration_months=_safe_int(entry.get("duration_months", 0)),
            is_current=bool(entry.get("is_current", False)),
            industry=str(entry.get("industry", "")),
            company_size=str(entry.get("company_size", "")),
            description=str(entry.get("description", "")),
        ))

    # Education
    education = []
    for entry in raw.get("education", []):
        if not isinstance(entry, dict):
            continue
        education.append(EducationEntry(
            institution=str(entry.get("institution", "")),
            degree=str(entry.get("degree", "")),
            field_of_study=str(entry.get("field_of_study", "")),
            start_year=entry.get("start_year"),
            end_year=entry.get("end_year"),
            grade=str(entry.get("grade", "")),
            tier=str(entry.get("tier", "")),
        ))

    # Skills
    skills = []
    for entry in raw.get("skills", []):
        if not isinstance(entry, dict):
            continue
        skills.append(SkillEntry(
            name=str(entry.get("name", "")),
            proficiency=str(entry.get("proficiency", "beginner")).lower(),
            endorsements=_safe_int(entry.get("endorsements", 0)),
            duration_months=_safe_int(entry.get("duration_months", 0)),
        ))

    # Redrob signals
    sig = raw.get("redrob_signals", {})
    salary_range = sig.get("expected_salary_range_inr_lpa", {})
    if not isinstance(salary_range, dict):
        salary_range = {}

    signals = RedrobSignals(
        profile_completeness_score=_safe_float(sig.get("profile_completeness_score", 0)),
        signup_date=sig.get("signup_date"),
        last_active_date=sig.get("last_active_date"),
        open_to_work_flag=bool(sig.get("open_to_work_flag", False)),
        profile_views_received_30d=_safe_int(sig.get("profile_views_received_30d", 0)),
        applications_submitted_30d=_safe_int(sig.get("applications_submitted_30d", 0)),
        recruiter_response_rate=_safe_float(sig.get("recruiter_response_rate", 0)),
        avg_response_time_hours=_safe_float(sig.get("avg_response_time_hours", 0)),
        skill_assessment_scores=sig.get("skill_assessment_scores", {}),
        connection_count=_safe_int(sig.get("connection_count", 0)),
        endorsements_received=_safe_int(sig.get("endorsements_received", 0)),
        notice_period_days=_safe_int(sig.get("notice_period_days", 90)),
        expected_salary_range_inr_lpa=salary_range,
        preferred_work_mode=str(sig.get("preferred_work_mode", "")),
        willing_to_relocate=bool(sig.get("willing_to_relocate", False)),
        github_activity_score=float(sig.get("github_activity_score", -1)),
        search_appearance_30d=_safe_int(sig.get("search_appearance_30d", 0)),
        saved_by_recruiters_30d=_safe_int(sig.get("saved_by_recruiters_30d", 0)),
        interview_completion_rate=_safe_float(sig.get("interview_completion_rate", 0)),
        offer_acceptance_rate=float(sig.get("offer_acceptance_rate", -1)),
        verified_email=bool(sig.get("verified_email", False)),
        verified_phone=bool(sig.get("verified_phone", False)),
        linkedin_connected=bool(sig.get("linkedin_connected", False)),
    )

    return Candidate(
        candidate_id=str(raw.get("candidate_id", "")),
        anonymized_name=str(profile.get("anonymized_name", "")),
        headline=str(profile.get("headline", "")),
        summary=str(profile.get("summary", "")),
        location=str(profile.get("location", "")),
        country=str(profile.get("country", "")),
        years_of_experience=_safe_float(profile.get("years_of_experience", 0)),
        current_title=str(profile.get("current_title", "")),
        current_company=str(profile.get("current_company", "")),
        current_company_size=str(profile.get("current_company_size", "")),
        current_industry=str(profile.get("current_industry", "")),
        career_history=career,
        education=education,
        skills=skills,
        certifications=raw.get("certifications", []),
        languages=raw.get("languages", []),
        redrob_signals=signals,
    )
