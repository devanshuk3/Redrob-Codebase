"""
Embedding generator — wraps SentenceTransformer for local CPU inference.
"""

import os
from typing import Dict, List, Optional

import numpy as np

from src.utils.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _configure_cpu_threads():
    """Configure optimal CPU threading for embedding inference."""
    import torch

    # Use physical cores (half of os.cpu_count() on hyperthreaded CPUs)
    # os.cpu_count() returns logical cores; for CPU inference, using physical
    # cores avoids cache thrashing from hyperthreading.
    logical_cores = os.cpu_count() or 4
    physical_cores = max(1, logical_cores // 2)

    torch.set_num_threads(physical_cores)
    torch.set_num_interop_threads(max(1, physical_cores // 2))

    # Also set via environment for BLAS/MKL backends
    os.environ.setdefault("OMP_NUM_THREADS", str(physical_cores))
    os.environ.setdefault("MKL_NUM_THREADS", str(physical_cores))

    logger.info(
        f"CPU threading: {physical_cores} threads "
        f"(from {logical_cores} logical cores)"
    )


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
        """Load the SentenceTransformer model (download if needed) strictly on CPU."""
        from sentence_transformers import SentenceTransformer

        # Configure CPU threading before model load
        _configure_cpu_threads()

        if os.path.exists(self.model_dir) and os.listdir(self.model_dir):
            logger.info(f"Loading model from local cache: {self.model_dir} (forcing CPU)")
            self.model = SentenceTransformer(self.model_dir, device="cpu")
        else:
            logger.info(f"Downloading model: {self.model_name} (forcing CPU)")
            self.model = SentenceTransformer(self.model_name, device="cpu")
            # Save to local cache for future offline runs
            os.makedirs(self.model_dir, exist_ok=True)
            self.model.save(self.model_dir)
            logger.info(f"Model saved to {self.model_dir}")


    def encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.

        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding. Defaults to Config.EMBEDDING_BATCH_SIZE.
            show_progress: Whether to show a progress bar (default: off for speed).

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

    def encode_concept_texts(self, concept_texts: dict) -> dict:
        """
        Encode a dict of concept texts into normalized embeddings.

        Args:
            concept_texts: {concept_name: text_description}

        Returns:
            {concept_name: np.ndarray} — normalized embeddings for each concept.
        """
        if self.model is None:
            self.load_model()

        result = {}
        for name, text in concept_texts.items():
            embedding = self.model.encode(
                [text],
                normalize_embeddings=True,
            )
            result[name] = embedding[0]
            logger.info(f"Encoded concept '{name}': shape {embedding[0].shape}")

        return result
