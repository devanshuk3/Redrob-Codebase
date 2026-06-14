"""
Tests for the ingestion module: data loading, schema validation,
candidate parsing, and honeypot detection.
"""

import json
import os
import tempfile

import pytest

# Ensure project root on path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.data_loader import load_candidates
from src.ingestion.schema_validator import validate_record
from src.ingestion.candidate_parser import parse_candidate, Candidate
from src.ingestion.honeypot_filter import compute_quality_score


# ── Fixtures ─────────────────────────────────────────────────────────

def _make_raw_candidate(**overrides):
    """Create a minimal valid raw candidate dict."""
    base = {
        "candidate_id": "CAND_TEST_001",
        "profile": {
            "anonymized_name": "Test User",
            "headline": "ML Engineer | 5+ years",
            "summary": "Experienced ML engineer.",
            "location": "Bangalore",
            "country": "India",
            "years_of_experience": 5.0,
            "current_title": "ML Engineer",
            "current_company": "TestCorp",
            "current_company_size": "1001-5000",
            "current_industry": "Technology",
        },
        "career_history": [
            {
                "company": "TestCorp",
                "title": "ML Engineer",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
                "industry": "Technology",
                "company_size": "1001-5000",
                "description": "Building ML systems for search and ranking.",
            }
        ],
        "education": [
            {
                "institution": "IIT Delhi",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2012,
                "end_year": 2016,
                "grade": "8.5 CGPA",
                "tier": "tier_1",
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 60},
            {"name": "Machine Learning", "proficiency": "advanced", "endorsements": 30, "duration_months": 48},
        ],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "native"}],
        "redrob_signals": {
            "profile_completeness_score": 85.0,
            "signup_date": "2023-01-01",
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 15,
            "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 12.0,
            "skill_assessment_scores": {},
            "connection_count": 200,
            "endorsements_received": 80,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20.0, "max": 35.0},
            "preferred_work_mode": "remote",
            "willing_to_relocate": True,
            "github_activity_score": 75.0,
            "search_appearance_30d": 50,
            "saved_by_recruiters_30d": 8,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    _deep_update(base, overrides)
    return base


def _deep_update(base, overrides):
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


# ── Schema Validator Tests ───────────────────────────────────────────

class TestSchemaValidator:
    def test_valid_record(self):
        record = _make_raw_candidate()
        is_valid, reason = validate_record(record)
        assert is_valid, f"Expected valid, got: {reason}"

    def test_missing_candidate_id(self):
        record = _make_raw_candidate()
        del record["candidate_id"]
        is_valid, reason = validate_record(record)
        assert not is_valid

    def test_missing_profile(self):
        record = _make_raw_candidate()
        del record["profile"]
        is_valid, reason = validate_record(record)
        assert not is_valid

    def test_missing_skills(self):
        record = _make_raw_candidate()
        del record["skills"]
        is_valid, reason = validate_record(record)
        assert not is_valid

    def test_non_numeric_experience(self):
        record = _make_raw_candidate()
        record["profile"]["years_of_experience"] = "not_a_number"
        is_valid, reason = validate_record(record)
        assert not is_valid

    def test_skills_not_list(self):
        record = _make_raw_candidate()
        record["skills"] = "python, java"
        is_valid, reason = validate_record(record)
        assert not is_valid

    def test_empty_candidate_id(self):
        record = _make_raw_candidate()
        record["candidate_id"] = ""
        is_valid, reason = validate_record(record)
        assert not is_valid


# ── Candidate Parser Tests ───────────────────────────────────────────

class TestCandidateParser:
    def test_basic_parsing(self):
        raw = _make_raw_candidate()
        candidate = parse_candidate(raw)
        assert isinstance(candidate, Candidate)
        assert candidate.candidate_id == "CAND_TEST_001"
        assert candidate.years_of_experience == 5.0
        assert candidate.current_title == "ML Engineer"

    def test_skills_parsed(self):
        raw = _make_raw_candidate()
        candidate = parse_candidate(raw)
        assert len(candidate.skills) == 2
        assert candidate.skills[0].name == "Python"
        assert candidate.skills[0].proficiency == "expert"

    def test_career_history_parsed(self):
        raw = _make_raw_candidate()
        candidate = parse_candidate(raw)
        assert len(candidate.career_history) == 1
        assert candidate.career_history[0].is_current is True

    def test_missing_optional_fields(self):
        raw = _make_raw_candidate()
        raw["certifications"] = []
        raw["career_history"] = []
        candidate = parse_candidate(raw)
        assert len(candidate.career_history) == 0
        assert len(candidate.certifications) == 0

    def test_redrob_signals_parsed(self):
        raw = _make_raw_candidate()
        candidate = parse_candidate(raw)
        assert candidate.redrob_signals.recruiter_response_rate == 0.8
        assert candidate.redrob_signals.github_activity_score == 75.0


# ── Honeypot Filter Tests ────────────────────────────────────────────

class TestHoneypotFilter:
    def test_normal_candidate_high_quality(self):
        raw = _make_raw_candidate()
        candidate = parse_candidate(raw)
        score = compute_quality_score(candidate)
        assert score > 0.5, f"Normal candidate should have high quality, got {score}"

    def test_skill_explosion_penalized(self):
        raw = _make_raw_candidate()
        # Add 65 skills — above MAX_SKILLS_REASONABLE
        raw["skills"] = [
            {"name": f"Skill{i}", "proficiency": "beginner", "endorsements": 0, "duration_months": 1}
            for i in range(65)
        ]
        candidate = parse_candidate(raw)
        score = compute_quality_score(candidate)
        assert score <= 0.9, f"Skill explosion should reduce quality, got {score}"

    def test_experience_timeline_inconsistency(self):
        raw = _make_raw_candidate()
        raw["profile"]["years_of_experience"] = 15.0  # graduated 2016, only 10 years ago
        candidate = parse_candidate(raw)
        score = compute_quality_score(candidate)
        assert score <= 0.9, f"Timeline inconsistency should be penalized, got {score}"

    def test_low_exp_high_salary(self):
        raw = _make_raw_candidate()
        raw["profile"]["years_of_experience"] = 1.0
        raw["redrob_signals"]["expected_salary_range_inr_lpa"] = {"min": 25.0, "max": 40.0}
        candidate = parse_candidate(raw)
        score = compute_quality_score(candidate)
        assert score < 0.95, f"Salary inconsistency should be penalized, got {score}"

    def test_short_career_many_skills(self):
        raw = _make_raw_candidate()
        raw["profile"]["years_of_experience"] = 0.5
        raw["skills"] = [
            {"name": f"Skill{i}", "proficiency": "beginner", "endorsements": 0, "duration_months": 1}
            for i in range(25)
        ]
        candidate = parse_candidate(raw)
        score = compute_quality_score(candidate)
        assert score < 0.9, f"Short career + many skills should be penalized, got {score}"


# ── Data Loader Tests ────────────────────────────────────────────────

class TestDataLoader:
    def test_load_with_limit(self):
        # Only works if candidates.jsonl exists
        if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "candidates.jsonl")):
            pytest.skip("candidates.jsonl not available")
        candidates = load_candidates(limit=10)
        assert len(candidates) == 10

    def test_load_from_temp_file(self):
        records = [
            json.dumps(_make_raw_candidate(candidate_id=f"CAND_{i:04d}"))
            for i in range(5)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(records))
            temp_path = f.name

        try:
            loaded = load_candidates(filepath=temp_path)
            assert len(loaded) == 5
        finally:
            os.unlink(temp_path)

    def test_malformed_json_skipped(self):
        lines = [
            json.dumps(_make_raw_candidate(candidate_id="CAND_001")),
            "this is not json {{{",
            json.dumps(_make_raw_candidate(candidate_id="CAND_003")),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(lines))
            temp_path = f.name

        try:
            loaded = load_candidates(filepath=temp_path)
            assert len(loaded) == 2  # malformed line skipped
        finally:
            os.unlink(temp_path)
