#!/usr/bin/env python3
"""
RedRob Candidate Ranking Pipeline — Single Entrypoint.

Pipeline flow (CHANGE 5 — pre-filtering):
  1. Load & validate candidates
  2. Honeypot quality filtering → ~50k-70k survivors
  3. Parse JD dynamically (CHANGE 1)
  4. Structured feature extraction → top 5k by structured score
  5. Semantic embedding & reranking on top 5k only
  6. Final score combination (CHANGE 7 — rebalanced)
  7. Top 100 selection
  8. Reasoning generation
  9. Submission CSV output + validation

Usage:
    python main.py
"""

import os
import sys
import time
from typing import Dict, List, Any

import numpy as np

import argparse

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import Config
from src.utils.logging_utils import get_logger
from src.ingestion.data_loader import load_candidates
from src.ingestion.schema_validator import validate_record
from src.ingestion.candidate_parser import Candidate, parse_candidate
from src.ingestion.honeypot_filter import compute_quality_score
from src.features.jd_feature_mapper import parse_job_description, JDFeatures
from src.features.feature_builder import build_structured_features
from src.semantic.text_builder import build_candidate_text, build_jd_text
from src.semantic.embedding_generator import EmbeddingGenerator
from src.semantic.embedding_store import EmbeddingStore
from src.semantic.semantic_ranker import compute_semantic_scores
from src.ranking.score_combiner import combine_scores
from src.ranking.ranker import rank_candidates
from src.ranking.reasoning_generator import generate_reasoning
from src.ranking.submission_builder import build_submission, save_full_scores

logger = get_logger(__name__)


def main():
    """Execute the full ranking pipeline."""
    parser = argparse.ArgumentParser(description="RedRob Candidate Ranking Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of candidates loaded")
    parser.add_argument("--jd", type=str, default=Config.JD_FILE, help="Path to job description file")
    parser.add_argument("--output", type=str, default=Config.SUBMISSION_FILE, help="Path to output CSV")
    args = parser.parse_args()

    pipeline_start = time.time()

    # ── 0. Setup ─────────────────────────────────────────────────────
    Config.seed_everything()
    Config.ensure_dirs()
    logger.info("=" * 70)
    logger.info("RedRob Candidate Ranking Pipeline — Starting")
    logger.info("=" * 70)

    # ── 1. Validate JD exists (CHANGE 1 — no hardcoded JD) ──────────
    jd_path = args.jd
    if not os.path.exists(jd_path):
        logger.error(
            f"Job description not found at {jd_path}. "
            f"Please place your job_description.md in the data/ directory."
        )
        sys.exit(1)

    # ── 2. Parse Job Description ─────────────────────────────────────
    logger.info("Phase 1: Parsing job description...")
    jd_features = parse_job_description(jd_path)
    jd_raw_text = open(jd_path, "r", encoding="utf-8").read()
    logger.info(f"  Must-have skills: {jd_features.must_have_skills[:10]}")
    logger.info(f"  Experience range: {jd_features.experience_range}")
    logger.info(f"  Domain keywords: {jd_features.domain_keywords[:10]}")

    # ── 3. Load & Validate Candidates ────────────────────────────────
    logger.info("Phase 2: Loading and validating candidates...")
    t0 = time.time()
    raw_records = load_candidates(limit=args.limit)

    valid_candidates: List[Candidate] = []
    invalid_count = 0

    for record in raw_records:
        is_valid, reason = validate_record(record)
        if not is_valid:
            invalid_count += 1
            continue
        candidate = parse_candidate(record)
        valid_candidates.append(candidate)

    logger.info(
        f"  Loaded {len(valid_candidates)} valid candidates "
        f"({invalid_count} rejected) in {time.time() - t0:.1f}s"
    )

    # ── 4. Quality Filtering (CHANGE 6 — strengthened honeypot) ──────
    logger.info("Phase 3: Quality filtering (honeypot detection)...")
    t0 = time.time()
    quality_scores: Dict[str, float] = {}
    filtered_candidates: List[Candidate] = []

    for candidate in valid_candidates:
        q_score = compute_quality_score(candidate)
        candidate.quality_score = q_score
        quality_scores[candidate.candidate_id] = q_score

        if q_score >= Config.QUALITY_FILTER_THRESHOLD:
            filtered_candidates.append(candidate)

    removed = len(valid_candidates) - len(filtered_candidates)
    logger.info(
        f"  Quality filter: {len(filtered_candidates)} passed, "
        f"{removed} removed in {time.time() - t0:.1f}s"
    )

    # ── 5. Structured Feature Extraction (CHANGE 8 — expanded) ───────
    logger.info("Phase 4: Structured feature extraction...")
    t0 = time.time()
    candidate_features: Dict[str, Dict[str, Any]] = {}

    for candidate in filtered_candidates:
        features = build_structured_features(candidate, jd_features)
        candidate_features[candidate.candidate_id] = features

    logger.info(f"  Extracted features for {len(candidate_features)} candidates in {time.time() - t0:.1f}s")

    # ── 6. Pre-Filter: Select Top K by Structured Score (CHANGE 5) ───
    logger.info(f"Phase 5: Pre-filtering to top {Config.STRUCTURED_TOP_K} by structured score...")
    sorted_by_structured = sorted(
        candidate_features.items(),
        key=lambda x: x[1]["structured_score"],
        reverse=True,
    )
    top_k_ids = set(cid for cid, _ in sorted_by_structured[:Config.STRUCTURED_TOP_K])
    top_k_candidates = [c for c in filtered_candidates if c.candidate_id in top_k_ids]

    logger.info(f"  Selected {len(top_k_candidates)} candidates for semantic reranking")

    # ── 7. Semantic Embedding & Similarity ───────────────────────────
    logger.info("Phase 6: Semantic embedding and similarity...")
    t0 = time.time()

    embedder = EmbeddingGenerator()
    store = EmbeddingStore()

    # Build candidate texts
    candidate_texts = [build_candidate_text(c) for c in top_k_candidates]
    candidate_ids_list = [c.candidate_id for c in top_k_candidates]

    # Try to load cached embeddings
    cached = store.load_candidate_embeddings(expected_count=len(top_k_candidates))

    if cached is not None:
        candidate_embeddings, cached_ids = cached
        # Verify IDs match
        if list(cached_ids) == candidate_ids_list:
            logger.info("  Using cached candidate embeddings")
        else:
            logger.info("  Cache IDs mismatch — regenerating embeddings")
            cached = None

    if cached is None:
        embedder.load_model()
        candidate_embeddings = embedder.encode_texts(candidate_texts)
        store.save_candidate_embeddings(
            candidate_embeddings,
            np.array(candidate_ids_list),
        )

    # JD embedding
    jd_text = build_jd_text(jd_raw_text)
    jd_embedding_cached = store.load_jd_embedding()

    if jd_embedding_cached is not None:
        jd_embedding = jd_embedding_cached
        logger.info("  Using cached JD embedding")
    else:
        if embedder.model is None:
            embedder.load_model()
        jd_embedding = embedder.encode_single(jd_text)
        store.save_jd_embedding(jd_embedding)

    # Compute semantic scores
    semantic_scores = compute_semantic_scores(
        jd_embedding, candidate_embeddings, candidate_ids_list
    )

    logger.info(f"  Semantic scoring complete in {time.time() - t0:.1f}s")

    # ── 8. Score Combination (CHANGE 7 — rebalanced) ─────────────────
    logger.info("Phase 7: Score combination...")

    # Only combine scores for candidates that went through semantic
    top_k_features = {cid: candidate_features[cid] for cid in top_k_ids}
    top_k_quality = {cid: quality_scores[cid] for cid in top_k_ids}

    scored_candidates = combine_scores(
        top_k_features, semantic_scores, top_k_quality
    )

    # ── 9. Rank & Select Top 100 ─────────────────────────────────────
    logger.info("Phase 8: Ranking and selection...")
    ranked = rank_candidates(scored_candidates, top_n=Config.TOP_N)

    # ── 10. Reasoning Generation (CHANGE 9 — template-based) ────────
    logger.info("Phase 9: Generating reasoning...")
    candidate_map = {c.candidate_id: c for c in top_k_candidates}
    ranked = generate_reasoning(ranked, candidate_map, jd_features)

    # ── 11. Build & Validate Submission ──────────────────────────────
    logger.info("Phase 10: Building submission CSV...")
    df = build_submission(ranked, output_path=args.output)

    # Also save full scores for analysis
    save_full_scores(scored_candidates)

    # ── Done ─────────────────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    logger.info("=" * 70)
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"Submission: {args.output}")
    logger.info(f"Top candidate: {ranked[0]['candidate_id']} "
                f"(score: {ranked[0]['final_score']:.4f})")
    logger.info(f"100th candidate: {ranked[-1]['candidate_id']} "
                f"(score: {ranked[-1]['final_score']:.4f})")
    logger.info("=" * 70)

    return df


if __name__ == "__main__":
    main()
