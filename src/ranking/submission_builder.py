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
