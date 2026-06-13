"""
Schema validation for candidate records.
Ensures required fields exist and have correct types.
"""

from typing import Any, Dict, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Required top-level keys
REQUIRED_KEYS = {"candidate_id", "profile", "skills", "redrob_signals"}

# Required profile keys
REQUIRED_PROFILE_KEYS = {"years_of_experience", "current_title"}


def validate_record(record: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate a single candidate record.

    Returns:
        (is_valid, reason) — True if the record passes validation.
    """
    if not isinstance(record, dict):
        return False, "Record is not a dictionary"

    # Check top-level keys
    missing = REQUIRED_KEYS - set(record.keys())
    if missing:
        return False, f"Missing required keys: {missing}"

    # Check candidate_id
    cid = record.get("candidate_id")
    if not cid or not isinstance(cid, str) or not cid.strip():
        return False, "Invalid or missing candidate_id"

    # Check profile is a dict
    profile = record.get("profile")
    if not isinstance(profile, dict):
        return False, "Profile is not a dictionary"

    # Check required profile fields
    yoe = profile.get("years_of_experience")
    if yoe is None:
        return False, "Missing years_of_experience"
    try:
        float(yoe)
    except (ValueError, TypeError):
        return False, f"years_of_experience is not numeric: {yoe}"

    # Skills should be a list
    skills = record.get("skills")
    if not isinstance(skills, list):
        return False, "Skills is not a list"

    # redrob_signals should be a dict
    signals = record.get("redrob_signals")
    if not isinstance(signals, dict):
        return False, "redrob_signals is not a dictionary"

    return True, "OK"
