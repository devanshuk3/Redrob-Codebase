"""
Submission builder — produces and validates the final submission.csv.
"""

from typing import Any, Dict, List

import pandas as pd

from src.utils.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_submission(
    ranked_candidates: List[Dict[str, Any]],
    output_path: str = None,
) -> pd.DataFrame:
    """
    Build and validate the submission CSV.

    Required columns: candidate_id, rank, score, reasoning

    Validates:
    - Exactly 100 rows
    - Unique ranks (1-100)
    - Unique candidate IDs
    - Scores are non-increasing
    - UTF-8 encoding

    Args:
        ranked_candidates: List of ranked dicts with reasoning.
        output_path: Path to save CSV. Defaults to Config.SUBMISSION_FILE.

    Returns:
        Validated DataFrame.
    """
    output_path = output_path or Config.SUBMISSION_FILE

    # Build DataFrame
    rows = []
    for entry in ranked_candidates:
        rows.append({
            "candidate_id": entry["candidate_id"],
            "rank": entry["rank"],
            "score": round(entry["final_score"], 6),
            "reasoning": entry.get("reasoning", ""),
        })

    df = pd.DataFrame(rows)

    # ── Validation ───────────────────────────────────────────────────
    errors = validate_submission(df)
    if errors:
        for err in errors:
            logger.error(f"Validation error: {err}")
        raise ValueError(f"Submission validation failed with {len(errors)} errors")

    # ── Save ─────────────────────────────────────────────────────────
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Submission saved: {output_path} ({len(df)} rows)")

    return df


def validate_submission(df: pd.DataFrame) -> List[str]:
    """
    Validate the submission DataFrame.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    # Exactly 100 rows
    if len(df) != Config.TOP_N:
        errors.append(f"Expected {Config.TOP_N} rows, got {len(df)}")

    # Required columns
    required_cols = {"candidate_id", "rank", "score", "reasoning"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
        return errors  # Can't validate further without required columns

    # Unique candidate IDs
    if df["candidate_id"].duplicated().any():
        dups = df[df["candidate_id"].duplicated()]["candidate_id"].tolist()
        errors.append(f"Duplicate candidate_ids: {dups}")

    # Unique ranks
    if df["rank"].duplicated().any():
        dups = df[df["rank"].duplicated()]["rank"].tolist()
        errors.append(f"Duplicate ranks: {dups}")

    # Ranks should be 1 to 100
    expected_ranks = set(range(1, Config.TOP_N + 1))
    actual_ranks = set(df["rank"].tolist())
    if actual_ranks != expected_ranks:
        errors.append(f"Ranks should be 1-{Config.TOP_N}, got {sorted(actual_ranks)[:5]}...")

    # Scores should be non-increasing (when sorted by rank)
    df_sorted = df.sort_values("rank")
    scores = df_sorted["score"].tolist()
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1] + 1e-9:
            errors.append(f"Score at rank {i + 1} ({scores[i]:.6f}) > rank {i} ({scores[i - 1]:.6f})")
            break

    # Reasoning should not be empty
    empty_reasoning = df["reasoning"].isna().sum() + (df["reasoning"] == "").sum()
    if empty_reasoning > 0:
        errors.append(f"{empty_reasoning} candidates have empty reasoning")

    return errors


def save_full_scores(
    all_scored: List[Dict[str, Any]],
    output_path: str = None,
):
    """Save all candidate scores (not just top 100) for analysis."""
    output_path = output_path or Config.CANDIDATE_SCORES_FILE
    df = pd.DataFrame(all_scored)
    df = df.sort_values("final_score", ascending=False)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Full scores saved: {output_path} ({len(df)} rows)")


def build_debug_csv(
    ranked_candidates: List[Dict[str, Any]],
    candidate_map: Dict[str, Any],
    output_path: str = None,
):
    """
    Build a comprehensive debug CSV containing scores, ranks,
    and all original candidate profile details to verify ranking logic.
    """
    from src.ingestion.candidate_parser import Candidate

    output_path = output_path or Config.DEBUG_FILE

    rows = []
    for entry in ranked_candidates:
        cid = entry["candidate_id"]
        cand: Candidate = candidate_map.get(cid)
        if not cand:
            continue

        # Extract structured details
        skills_str = ", ".join([s.name for s in cand.skills])
        education_str = "; ".join([
            f"{e.degree} in {e.field_of_study} from {e.institution} ({e.start_year}-{e.end_year})"
            for e in cand.education
        ])
        career_str = "; ".join([
            f"{j.title} at {j.company} ({j.duration_months} mos, current={j.is_current})"
            for j in cand.career_history
        ])
        cert_names = []
        for cert in cand.certifications:
            if isinstance(cert, dict) and "name" in cert:
                cert_names.append(cert["name"])
            elif isinstance(cert, str):
                cert_names.append(cert)
        certs_str = ", ".join(cert_names)

        langs_str = ", ".join([
            f"{lang.get('name', '')} ({lang.get('proficiency', '')})"
            if isinstance(lang, dict) else str(lang)
            for lang in cand.languages
        ])

        sig = cand.redrob_signals
        sal_min = sig.expected_salary_range_inr_lpa.get("min", 0.0)
        sal_max = sig.expected_salary_range_inr_lpa.get("max", 0.0)
        salary_str = f"{sal_min}-{sal_max} LPA"

        row = {
            # Final Results
            "rank": entry.get("rank"),
            "candidate_id": cid,
            "final_score": round(entry.get("final_score", 0.0), 6),
            "reasoning": entry.get("reasoning", ""),

            # Component Scores
            "semantic_score": round(entry.get("semantic_score", 0.0), 6),
            "structured_score": round(entry.get("structured_score", 0.0), 6),
            "behavioral_score": round(entry.get("behavioral_score", 0.0), 6),
            "quality_score": round(entry.get("quality_score", 0.0), 6),

            # Sub-Scores (if available)
            "skills_score": round(entry.get("skills_score", 0.0), 6),
            "experience_score": round(entry.get("experience_score", 0.0), 6),
            "education_score": round(entry.get("education_score", 0.0), 6),
            "certification_score": round(entry.get("certification_score", 0.0), 6),

            # Original Profile Details
            "anonymized_name": cand.anonymized_name,
            "headline": cand.headline,
            "summary": cand.summary,
            "location": cand.location,
            "country": cand.country,
            "years_of_experience": cand.years_of_experience,
            "current_title": cand.current_title,
            "current_company": cand.current_company,
            "current_company_size": cand.current_company_size,
            "current_industry": cand.current_industry,
            "skills": skills_str,
            "education": education_str,
            "career_history": career_str,
            "certifications": certs_str,
            "languages": langs_str,

            # Behavioral / Telemetry Signals
            "profile_completeness_score": sig.profile_completeness_score,
            "signup_date": sig.signup_date,
            "last_active_date": sig.last_active_date,
            "open_to_work": sig.open_to_work_flag,
            "profile_views_received_30d": sig.profile_views_received_30d,
            "applications_submitted_30d": sig.applications_submitted_30d,
            "recruiter_response_rate": sig.recruiter_response_rate,
            "avg_response_time_hours": sig.avg_response_time_hours,
            "connection_count": sig.connection_count,
            "endorsements_received": sig.endorsements_received,
            "notice_period_days": sig.notice_period_days,
            "expected_salary": salary_str,
            "preferred_work_mode": sig.preferred_work_mode,
            "willing_to_relocate": sig.willing_to_relocate,
            "github_activity_score": sig.github_activity_score,
            "search_appearance_30d": sig.search_appearance_30d,
            "saved_by_recruiters_30d": sig.saved_by_recruiters_30d,
            "interview_completion_rate": sig.interview_completion_rate,
            "offer_acceptance_rate": sig.offer_acceptance_rate,
            "verified_email": sig.verified_email,
            "verified_phone": sig.verified_phone,
            "linkedin_connected": sig.linkedin_connected,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Debug candidate CSV saved: {output_path} ({len(df)} rows)")
