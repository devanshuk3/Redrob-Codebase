"""
Centralized configuration for the RedRob candidate ranking pipeline.
All tunable parameters in one place for easy experimentation.
"""

import os
import random
import numpy as np


class Config:
    """Pipeline configuration — all weights, thresholds, and paths."""

    # Reproducibility
    RANDOM_SEED = 42

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

    CANDIDATES_FILE = os.path.join(BASE_DIR, "candidates.jsonl")
    JD_FILE = os.path.join(BASE_DIR, "job_description.md")
    SUBMISSION_FILE = os.path.join(BASE_DIR, "submission.csv")
    CANDIDATE_SCORES_FILE = os.path.join(OUTPUTS_DIR, "candidate_scores.csv")
    DEBUG_DIR = os.path.join(OUTPUTS_DIR, "debug")
    DEBUG_FILE = os.path.join(DEBUG_DIR, "candidate_debug.csv")

    EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "candidate_embeddings.npy")
    EMBEDDING_IDS_FILE = os.path.join(MODELS_DIR, "candidate_ids.npy")
    JD_EMBEDDING_FILE = os.path.join(MODELS_DIR, "jd_embedding.npy")

    # Model
    SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
    SENTENCE_MODEL_DIR = os.path.join(MODELS_DIR, "all-MiniLM-L6-v2")
    EMBEDDING_BATCH_SIZE = 512

    # Cross-Encoder Reranking (top-300 stage — joint JD+candidate relevance)
    CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CROSS_ENCODER_MODEL_DIR = os.path.join(MODELS_DIR, "ms-marco-MiniLM-L-6-v2")
    CROSS_ENCODER_BATCH_SIZE = 32
    CROSS_ENCODER_MAX_LENGTH = 512          # tokens, total for the pair
    CROSS_ENCODER_TEXT_MAX_CHARS = 800      # per side (JD, candidate) — keeps the
                                             # pair comfortably under the token limit
    CROSS_ENCODER_MODIFIER_STRENGTH = 0.15  # placeholder multiplicative weight —
                                             # replace with a learned-fusion feature
                                             # in score_combiner.py once that's ready
    # Final Score Weights (TASK 11 — rebalanced)
    # technical_fit=60%, experience_fit=20%, behavioral_fit=15%, hiring_fit=5%
    # semantic ≈ holistic technical signal, structured = granular technical+experience
    # behavioral = hiring readiness, quality = profile trust
    WEIGHT_LEXICAL = 0.10   
    WEIGHT_SEMANTIC = 0.15        # semantic similarity to JD
    WEIGHT_STRUCTURED = 0.50      # detailed technical + experience scoring
    WEIGHT_BEHAVIORAL = 0.15      # hiring readiness signals
    WEIGHT_QUALITY = 0.10         # profile trustworthiness

    # Structured Sub-Score Weights (TASK 11 — rebalanced)
    # Technical relevance must dominate within structured scoring.
    STRUCT_WEIGHT_SKILL = 0.20         # JD skill matching
    STRUCT_WEIGHT_RETRIEVAL = 0.20     # retrieval/search domain
    STRUCT_WEIGHT_RANKING = 0.15       # ranking/recommendation domain
    STRUCT_WEIGHT_EVALUATION = 0.10    # evaluation framework experience
    STRUCT_WEIGHT_PRODUCTION = 0.15    # production deployment experience
    STRUCT_WEIGHT_EXPERIENCE = 0.10    # years + role relevance
    STRUCT_WEIGHT_EDUCATION = 0.05     # education (de-prioritized)
    STRUCT_WEIGHT_CERTIFICATION = 0.05 # certifications (de-prioritized)

    # Pre-filter Thresholds (CHANGE 5)
    QUALITY_FILTER_THRESHOLD = 0.20       # drop candidates below this quality
    STRUCTURED_TOP_K = 5000               # top-K after structured scoring → semantic

    # Honeypot Thresholds (CHANGE 6)
    MAX_SKILLS_REASONABLE = 60
    MAX_SALARY_FOR_LOW_EXP = 30.0         # LPA — unreasonable for <2 yr exp
    LOW_EXP_THRESHOLD_YEARS = 2.0
    MIN_QUALITY_FOR_INCLUSION = 0.20

    # Experience Ideal Range
    EXPERIENCE_IDEAL_MIN = 5
    EXPERIENCE_IDEAL_MAX = 9

    # Ranking Output
    TOP_N = 100

    # Fuzzy Matching
    FUZZY_MATCH_THRESHOLD = 80

    # LLM Hype Penalty (TASK 3)
    LLM_HYPE_PENALTY_MAX = 0.12           # maximum penalty for hype-heavy profiles

    # Notice Period Penalty (TASK 9)
    NOTICE_PERIOD_PENALTY_MAX = 0.05      # max deduction for long notice

    # Relocation Bonus (TASK 10)
    RELOCATION_BONUS = 0.02               # small bonus for willing to relocate

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
        os.makedirs(cls.DEBUG_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
