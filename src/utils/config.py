"""
Centralized configuration for the RedRob candidate ranking pipeline.
All tunable parameters in one place for easy experimentation.
"""

import os
import random
import numpy as np


class Config:
    """Pipeline configuration — all weights, thresholds, and paths."""

    # ── Reproducibility ──────────────────────────────────────────────
    RANDOM_SEED = 42

    # ── Paths ────────────────────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

    CANDIDATES_FILE = os.path.join(DATA_DIR, "candidates.jsonl")
    JD_FILE = os.path.join(DATA_DIR, "job_description.md")
    SUBMISSION_FILE = os.path.join(OUTPUTS_DIR, "submission.csv")
    CANDIDATE_SCORES_FILE = os.path.join(OUTPUTS_DIR, "candidate_scores.csv")

    EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "candidate_embeddings.npy")
    EMBEDDING_IDS_FILE = os.path.join(MODELS_DIR, "candidate_ids.npy")
    JD_EMBEDDING_FILE = os.path.join(MODELS_DIR, "jd_embedding.npy")

    # ── Model ────────────────────────────────────────────────────────
    SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
    SENTENCE_MODEL_DIR = os.path.join(MODELS_DIR, "all-MiniLM-L6-v2")
    EMBEDDING_BATCH_SIZE = 512

    # ── Final Score Weights (CHANGE 7 — rebalanced) ──────────────────
    # These four groups sum to 1.0
    WEIGHT_SEMANTIC = 0.25
    WEIGHT_STRUCTURED = 0.45
    WEIGHT_BEHAVIORAL = 0.20
    WEIGHT_QUALITY = 0.10

    # ── Structured Sub-Score Weights (CHANGE 8 — expanded) ───────────
    # Weights within the structured component; these are relative and
    # will be normalized to sum to 1.0 internally.
    STRUCT_WEIGHT_SKILL = 0.20
    STRUCT_WEIGHT_RETRIEVAL = 0.20
    STRUCT_WEIGHT_RANKING = 0.15
    STRUCT_WEIGHT_EVALUATION = 0.10
    STRUCT_WEIGHT_PRODUCTION = 0.15
    STRUCT_WEIGHT_EXPERIENCE = 0.10
    STRUCT_WEIGHT_EDUCATION = 0.05       # CHANGE 4 — de-prioritized
    STRUCT_WEIGHT_CERTIFICATION = 0.05   # CHANGE 4 — de-prioritized

    # ── Pre-filter Thresholds (CHANGE 5) ─────────────────────────────
    QUALITY_FILTER_THRESHOLD = 0.20       # drop candidates below this quality
    STRUCTURED_TOP_K = 5000               # top-K after structured scoring → semantic

    # ── Honeypot Thresholds (CHANGE 6) ───────────────────────────────
    MAX_SKILLS_REASONABLE = 60
    MAX_SALARY_FOR_LOW_EXP = 30.0         # LPA — unreasonable for <2 yr exp
    LOW_EXP_THRESHOLD_YEARS = 2.0
    MIN_QUALITY_FOR_INCLUSION = 0.20

    # ── Experience Ideal Range ───────────────────────────────────────
    EXPERIENCE_IDEAL_MIN = 5
    EXPERIENCE_IDEAL_MAX = 9

    # ── Ranking Output ───────────────────────────────────────────────
    TOP_N = 100

    # ── Fuzzy Matching ───────────────────────────────────────────────
    FUZZY_MATCH_THRESHOLD = 80

    @classmethod
    def seed_everything(cls):
        """Fix all random seeds for deterministic execution."""
        random.seed(cls.RANDOM_SEED)
        np.random.seed(cls.RANDOM_SEED)

    @classmethod
    def ensure_dirs(cls):
        """Create output directories if they don't exist."""
        os.makedirs(cls.OUTPUTS_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
