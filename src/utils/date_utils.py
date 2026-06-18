"""
Date parsing and calculation utilities.
"""

from datetime import datetime, date
from typing import Optional


def parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse a date string (YYYY-MM-DD) into a date object."""
    if not date_str:
        return None
    try:
        s = str(date_str).strip()
        return date(int(s[0:4]), int(s[5:7]), int(s[8:10]))
    except (ValueError, TypeError, IndexError):
        return None


def years_between(start: Optional[date], end: Optional[date]) -> float:
    """Calculate years between two dates. Uses today if end is None."""
    if not start:
        return 0.0
    if not end:
        end = date.today()
    delta = end - start
    return max(0.0, delta.days / 365.25)


def years_since_graduation(end_year: Optional[int]) -> float:
    """Years since graduation year to now."""
    if not end_year:
        return 0.0
    try:
        end_year = int(end_year)
        return max(0.0, datetime.now().year - end_year)
    except (ValueError, TypeError):
        return 0.0
