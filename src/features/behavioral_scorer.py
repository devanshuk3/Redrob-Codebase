"""
Behavioral signal scorer — maps redrob_signals to a composite score.
TASK 9 — notice period as hiring modifier.
TASK 10 — relocation bonus.
"""

from datetime import date

from src.ingestion.candidate_parser import Candidate
from src.utils.config import Config
from src.utils.date_utils import parse_date


def score_behavioral(candidate: Candidate) -> float:
    """
    Score behavioral signals from redrob platform data.

    Sub-signals (weighted):
    - open_to_work (15%)
    - recruiter_response_rate (18%)
    - last_active recency (18%)
    - github_activity (10%)
    - saved_by_recruiters (6%)
    - interview_completion (10%)
    - profile_completeness (8%)
    - relocation + notice as modifiers (8%)
    - availability combo penalty (7%)

    Returns:
        Normalized score [0, 1].
    """
    sig = candidate.redrob_signals

    # 1. Open to work — stronger weight per JD's emphasis on availability
    open_score = 1.0 if sig.open_to_work_flag else 0.2

    # 2. Recruiter response rate
    resp_score = min(1.0, sig.recruiter_response_rate)

    # 3. Last active recency
    last_active = parse_date(sig.last_active_date)
    if last_active:
        ref_date = max(date.today(), date(2026, 6, 18))
        days_ago = (ref_date - last_active).days
        if days_ago <= 30:
            recency_score = 1.0
        elif days_ago <= 60:
            recency_score = 0.85
        elif days_ago <= 90:
            recency_score = 0.7
        elif days_ago <= 150:
            recency_score = 0.4
        elif days_ago <= 365:
            recency_score = 0.2
        else:
            recency_score = 0.05
    else:
        recency_score = 0.15

    # 4. GitHub activity
    github = sig.github_activity_score
    if github < 0:
        github_score = 0.2  # missing data, small default
    else:
        github_score = min(1.0, github / 100.0)

    # 5. Saved by recruiters
    saved = sig.saved_by_recruiters_30d
    if saved >= 10:
        saved_score = 1.0
    elif saved >= 5:
        saved_score = 0.7
    elif saved >= 1:
        saved_score = 0.4
    else:
        saved_score = 0.1

    # 6. Interview completion rate
    interview_score = min(1.0, sig.interview_completion_rate)

    # 7. Profile completeness
    completeness_score = min(1.0, sig.profile_completeness_score / 100.0)

    # 8. Notice period penalty (TASK 9)
    # Hiring modifier: shorter notice = better availability
    notice = sig.notice_period_days
    if notice <= 0:
        notice_score = 0.5  # unknown
    elif notice <= 30:
        notice_score = 1.0   # no penalty
    elif notice <= 60:
        notice_score = 0.85  # small penalty
    elif notice <= 90:
        notice_score = 0.6   # moderate penalty
    else:
        notice_score = 0.3   # strong penalty (120+ days)

    # 9. Relocation bonus (TASK 10)
    relocation_score = 1.0 if sig.willing_to_relocate else 0.5

    # 10. Availability combo penalty (Fix #4) — JD: "a perfect-on-paper candidate
    # who hasn't logged in for 6 months and has a 5% response rate is not actually available"
    availability_combo = 1.0
    is_stale = last_active is not None and (max(date.today(), date(2026, 6, 18)) - last_active).days > 90
    is_unresponsive = sig.recruiter_response_rate < 0.15
    not_seeking = not sig.open_to_work_flag

    concern_count = sum([is_stale, is_unresponsive, not_seeking])
    if concern_count >= 3:
        availability_combo = 0.1  # all three: very low
    elif concern_count == 2:
        availability_combo = 0.3  # two concerns: significant penalty
    elif concern_count == 1:
        availability_combo = 0.7  # one concern: mild penalty

    # Weighted combination — availability signals get more weight per JD
    score = (
        0.15 * open_score
        + 0.18 * resp_score
        + 0.18 * recency_score
        + 0.10 * github_score
        + 0.06 * saved_score
        + 0.10 * interview_score
        + 0.08 * completeness_score
        + 0.04 * notice_score
        + 0.04 * relocation_score
        + 0.07 * availability_combo
    )

    return round(min(1.0, score), 4)
