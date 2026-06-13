"""
File I/O helpers — JSON line reading, CSV writing, numpy persistence.
"""

import csv
import json
import os
from typing import Any, Dict, Generator, List, Optional

import numpy as np

from .logging_utils import get_logger

logger = get_logger(__name__)


def stream_jsonl(filepath: str) -> Generator[Dict[str, Any], None, None]:
    """
    Stream-read a JSONL file line by line (memory efficient).
    Skips and logs malformed lines.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSONL file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Malformed JSON at line {line_num}: {e}")
                continue


def read_text_file(filepath: str) -> str:
    """Read an entire text file and return its contents."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Text file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def write_csv(filepath: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    """Write a list of dicts to a CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {filepath}")


def save_numpy(filepath: str, arr: np.ndarray):
    """Save a numpy array to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, arr)
    logger.info(f"Saved numpy array {arr.shape} to {filepath}")


def load_numpy(filepath: str) -> Optional[np.ndarray]:
    """Load a numpy array from disk, or None if it doesn't exist."""
    if not os.path.exists(filepath):
        return None
    arr = np.load(filepath, allow_pickle=True)
    logger.info(f"Loaded numpy array {arr.shape} from {filepath}")
    return arr
