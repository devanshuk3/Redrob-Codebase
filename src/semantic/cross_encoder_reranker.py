"""
Cross-encoder reranker — joint JD+candidate relevance scoring on a small
candidate pool (the top ~300 by combined score), as a more accurate
complement to the cheap bi-encoder cosine similarity computed earlier.

Why this only runs on 300, not 5,000 or 100,000:
A bi-encoder (see embedding_generator.py) embeds the JD and each candidate
independently, so every candidate's embedding can be computed once and
cached — comparison is then just a cosine similarity. A cross-encoder
jointly encodes (JD, candidate) as a single input through shared attention
layers, which is more accurate but cannot be cached or precomputed per
candidate: every pair needs its own forward pass. That's only affordable
on a small, already-narrowed pool.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.ingestion.candidate_parser import Candidate
from src.semantic.text_builder import build_candidate_text, build_jd_text
from src.utils.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Map raw cross-encoder logits to [0, 1]."""
    return 1.0 / (1.0 + np.exp(-x))


class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder for local CPU inference.
    Mirrors EmbeddingGenerator's local-cache-first loading pattern: looks
    for a locally cached copy of the model first, downloads only if missing.
    """

    def __init__(self, model_name: Optional[str] = None, model_dir: Optional[str] = None):
        self.model_name = model_name or Config.CROSS_ENCODER_MODEL_NAME
        self.model_dir = model_dir or Config.CROSS_ENCODER_MODEL_DIR
        self.model = None

    def load_model(self):
        """Load the CrossEncoder model (download if needed) strictly on CPU."""
        from sentence_transformers import CrossEncoder

        if os.path.exists(self.model_dir) and os.listdir(self.model_dir):
            logger.info(f"Loading cross-encoder from local cache: {self.model_dir} (forcing CPU)")
            self.model = CrossEncoder(
                self.model_dir, device="cpu", max_length=Config.CROSS_ENCODER_MAX_LENGTH,
            )
        else:
            logger.info(f"Local cross-encoder cache not found — downloading '{self.model_name}'")
            self.model = CrossEncoder(
                self.model_name, device="cpu", max_length=Config.CROSS_ENCODER_MAX_LENGTH,
            )

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Score (jd_text, candidate_text) pairs.

        Returns:
            Sigmoid-normalized scores in [0, 1], same order as `pairs`.
        """
        if not pairs:
            return np.array([])
        if self.model is None:
            self.load_model()
        raw_scores = self.model.predict(
            pairs, batch_size=Config.CROSS_ENCODER_BATCH_SIZE, show_progress_bar=False,
        )
        return _sigmoid(np.asarray(raw_scores, dtype=np.float64))


def compute_cross_encoder_scores(
    jd_raw_text: str,
    candidates: List[Candidate],
    reranker: Optional[CrossEncoderReranker] = None,
) -> Dict[str, float]:
    """
    Compute cross-encoder relevance scores for a small candidate pool.

    Args:
        jd_raw_text: Raw job description text (same input used elsewhere
            for build_jd_text — pass the JD file's full text, not a path).
        candidates: Candidates to score. Keep this small (≤300) — a
            cross-encoder forward pass can't be cached per-candidate the
            way the bi-encoder embeddings are.
        reranker: Optional pre-loaded CrossEncoderReranker, so callers can
            reuse a loaded model across multiple calls instead of reloading.

    Returns:
        {candidate_id: cross_encoder_score} — sigmoid-normalized to [0, 1].
    """
    if not candidates:
        return {}

    reranker = reranker or CrossEncoderReranker()

    # Cross-encoder pairs are jointly encoded into ONE sequence, so keep each
    # side shorter than the bi-encoder's 1500-char default. Otherwise the
    # tokenizer silently truncates the pair, usually losing the candidate's
    # tail content rather than raising an error.
    jd_text = build_jd_text(jd_raw_text, max_chars=Config.CROSS_ENCODER_TEXT_MAX_CHARS)
    pairs = [
        (jd_text, build_candidate_text(c, max_chars=Config.CROSS_ENCODER_TEXT_MAX_CHARS))
        for c in candidates
    ]
    candidate_ids = [c.candidate_id for c in candidates]

    scores = reranker.score_pairs(pairs)
    result = {cid: round(float(s), 4) for cid, s in zip(candidate_ids, scores)}

    logger.info(
        f"Computed cross-encoder scores for {len(result)} candidates "
        f"(mean={np.mean(scores):.4f}, max={np.max(scores):.4f})"
    )
    return result
