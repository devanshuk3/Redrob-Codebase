"""
Streaming data loader for candidates.jsonl.
Memory-efficient line-by-line reading.
"""

from typing import Dict, Any, List, Optional

from src.utils.config import Config
from src.utils.io_utils import stream_jsonl
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_candidates(
    filepath: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load candidate records from a JSONL file.

    Args:
        filepath: Path to the JSONL file. Defaults to Config.CANDIDATES_FILE.
        limit: Maximum number of records to load (for testing). None = all.

    Returns:
        List of raw candidate dictionaries.
    """
    filepath = filepath or Config.CANDIDATES_FILE
    logger.info(f"Loading candidates from {filepath}...")

    candidates = []
    for i, record in enumerate(stream_jsonl(filepath)):
        if limit is not None and i >= limit:
            break
        candidates.append(record)

    logger.info(f"Loaded {len(candidates)} candidate records")
    return candidates
