"""
Behavioral signal scorer — maps redrob_signals to a composite score.
"""

from datetime import date

from src.ingestion.candidate_parser import Candidate
from src.utils.date_utils import parse_date


def score_behavioral(candidate: Candidate) -> float:
    """
    Score behavioral signals from redrob platform data.

    Sub-signals (weighted):
    - open_to_work (10%)
    - recruiter_response_rate (15%)
    - last_active recency (15%)
    - github_activity (15%)
    - saved_by_recruiters (10%)
    - interview_completion (15%)
    - notice_period (10%)
    - profile_completeness (10%)

    Returns:
        Normalized score [0, 1].
    """
    sig = candidate.redrob_signals

    # ── 1. Open to work ──────────────────────────────────────────────
    open_score = 1.0 if sig.open_to_work_flag else 0.3

    # ── 2. Recruiter response rate ───────────────────────────────────
    resp_score = min(1.0, sig.recruiter_response_rate)

    # ── 3. Last active recency ───────────────────────────────────────
    last_active = parse_date(sig.last_active_date)
    if last_active:
        days_ago = (date.today() - last_active).days
        if days_ago <= 30:
            recency_score = 1.0
        elif days_ago <= 90:
            recency_score = 0.8
        elif days_ago <= 180:
            recency_score = 0.5
        elif days_ago <= 365:
            recency_score = 0.3
        else:
            recency_score = 0.1
    else:
        recency_score = 0.2

    # ── 4. GitHub activity ───────────────────────────────────────────
    github = sig.github_activity_score
    if github < 0:
        github_score = 0.2  # missing data, small default
    else:
        github_score = min(1.0, github / 100.0)

    # ── 5. Saved by recruiters ───────────────────────────────────────
    saved = sig.saved_by_recruiters_30d
    if saved >= 10:
        saved_score = 1.0
    elif saved >= 5:
        saved_score = 0.7
    elif saved >= 1:
        saved_score = 0.4
    else:
        saved_score = 0.1

    # ── 6. Interview completion rate ─────────────────────────────────
    interview_score = min(1.0, sig.interview_completion_rate)

    # ── 7. Notice period (shorter = better, cap at 90 days) ──────────
    notice = sig.notice_period_days
    if notice <= 0:
        notice_score = 0.5  # unknown
    elif notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.7
    elif notice <= 90:
        notice_score = 0.4
    else:
        notice_score = 0.2

    # ── 8. Profile completeness ──────────────────────────────────────
    completeness_score = min(1.0, sig.profile_completeness_score / 100.0)

    # ── Weighted combination ─────────────────────────────────────────
    score = (
        0.10 * open_score
        + 0.15 * resp_score
        + 0.15 * recency_score
        + 0.15 * github_score
        + 0.10 * saved_score
        + 0.15 * interview_score
        + 0.10 * notice_score
        + 0.10 * completeness_score
    )

    return round(min(1.0, score), 4)
