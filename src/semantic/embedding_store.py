"""
Embedding persistence — save/load precomputed embeddings to avoid recomputation.
"""

from typing import Optional, Tuple

import numpy as np

from src.utils.config import Config
from src.utils.io_utils import save_numpy, load_numpy
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class EmbeddingStore:
    """Manages cached embeddings on disk."""

    def __init__(
        self,
        embeddings_path: Optional[str] = None,
        ids_path: Optional[str] = None,
        jd_path: Optional[str] = None,
    ):
        self.embeddings_path = embeddings_path or Config.EMBEDDINGS_FILE
        self.ids_path = ids_path or Config.EMBEDDING_IDS_FILE
        self.jd_path = jd_path or Config.JD_EMBEDDING_FILE

    def save_candidate_embeddings(
        self,
        embeddings: np.ndarray,
        candidate_ids: np.ndarray,
    ):
        """Save candidate embeddings and their IDs to disk."""
        save_numpy(self.embeddings_path, embeddings)
        save_numpy(self.ids_path, candidate_ids)

    def load_candidate_embeddings(
        self,
        expected_count: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load cached candidate embeddings if they exist and match expected count.

        Returns:
            Tuple of (embeddings, candidate_ids) or None if cache is invalid.
        """
        embeddings = load_numpy(self.embeddings_path)
        ids = load_numpy(self.ids_path)

        if embeddings is None or ids is None:
            logger.info("No cached embeddings found")
            return None

        if len(embeddings) != len(ids):
            logger.warning("Embedding/ID count mismatch — invalidating cache")
            return None

        if expected_count is not None and len(embeddings) != expected_count:
            logger.warning(
                f"Cached embeddings count ({len(embeddings)}) != expected ({expected_count}) "
                f"— invalidating cache"
            )
            return None

        logger.info(f"Loaded {len(embeddings)} cached embeddings")
        return embeddings, ids

    def save_jd_embedding(self, embedding: np.ndarray):
        """Save JD embedding to disk."""
        save_numpy(self.jd_path, embedding)

    def load_jd_embedding(self) -> Optional[np.ndarray]:
        """Load cached JD embedding."""
        return load_numpy(self.jd_path)
