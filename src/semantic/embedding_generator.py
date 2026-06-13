"""
Embedding generator — wraps SentenceTransformer for local CPU inference.
"""

from typing import List, Optional

import numpy as np

from src.utils.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using a local SentenceTransformer model.
    No external API calls — fully offline after initial model download.
    """

    def __init__(self, model_name: Optional[str] = None, model_dir: Optional[str] = None):
        self.model_name = model_name or Config.SENTENCE_MODEL_NAME
        self.model_dir = model_dir or Config.SENTENCE_MODEL_DIR
        self.model = None

    def load_model(self):
        """Load the SentenceTransformer model (download if needed)."""
        from sentence_transformers import SentenceTransformer
        import os

        if os.path.exists(self.model_dir) and os.listdir(self.model_dir):
            logger.info(f"Loading model from local cache: {self.model_dir}")
            self.model = SentenceTransformer(self.model_dir)
        else:
            logger.info(f"Downloading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            # Save to local cache for future offline runs
            os.makedirs(self.model_dir, exist_ok=True)
            self.model.save(self.model_dir)
            logger.info(f"Model saved to {self.model_dir}")

    def encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.

        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding. Defaults to Config.EMBEDDING_BATCH_SIZE.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        if self.model is None:
            self.load_model()

        batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
        logger.info(f"Encoding {len(texts)} texts in batches of {batch_size}")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2-normalize for cosine similarity
        )

        logger.info(f"Generated embeddings: shape {embeddings.shape}")
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        if self.model is None:
            self.load_model()
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
        )
        return embedding[0]
