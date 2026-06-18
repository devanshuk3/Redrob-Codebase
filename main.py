# !/usr/bin/env python3
"""
RedRob Candidate Ranking Pipeline — Single Entrypoint.

Pipeline flow (MERGED — notebook improvements integrated):
  1. Load & validate candidates
  2. Fast pre-filter — cheap checks (non-tech role, YOE hard minimum)
  3. Parse JD dynamically
  4. Structured feature extraction → top 5k by structured score
  5. Semantic embedding & reranking on top 5k only
  6. Concept embedding computation & scoring
  7. Final score combination (with concept scores)
  8. Detailed honeypot on top 300 only
  9. Top 100 selection
  10. Reasoning generation
  11. Submission CSV output + validation

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

import heapq
from src.utils.io_utils import stream_jsonl
from src.utils.config import Config
from src.utils.logging_utils import get_logger
from src.ingestion.data_loader import load_candidates
from src.ingestion.schema_validator import validate_record
from src.ingestion.candidate_parser import Candidate, parse_candidate
from src.ingestion.honeypot_filter import compute_quality_score, _check_non_tech_role, fast_prefilter_raw
from src.features.jd_feature_mapper import parse_job_description, JDFeatures
from src.features.feature_builder import build_structured_features
from src.semantic.text_builder import build_candidate_text, build_jd_text
from src.semantic.embedding_generator import EmbeddingGenerator
from src.semantic.embedding_store import EmbeddingStore
from src.semantic.semantic_ranker import compute_semantic_scores, compute_concept_scores
from src.ranking.score_combiner import combine_scores
from src.ranking.ranker import rank_candidates
from src.ranking.reasoning_generator import generate_reasoning
from src.ranking.submission_builder import build_submission, save_full_scores, build_debug_csv
from src.utils.constants import (
    RANKING_CONCEPT_TEXT, EVALUATION_CONCEPT_TEXT, PRODUCTION_CONCEPT_TEXT,
)

logger = get_logger(__name__)


def fast_prefilter(candidate: Candidate, min_yoe: float = 3.0) -> bool:
    """
    Returns True if candidate passes fast pre-filter (should be kept).

    Cheap checks only:
    - Not a non-tech role
    - Meets minimum years of experience threshold
    """
    if _check_non_tech_role(candidate):
        return False
    if candidate.years_of_experience < min_yoe:
        return False
    return True


def main():
    """Execute the full ranking pipeline."""
    parser = argparse.ArgumentParser(description="RedRob Candidate Ranking Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of candidates loaded")
    parser.add_argument("--jd", type=str, default=Config.JD_FILE, help="Path to job description file")
    parser.add_argument("--output", type=str, default=Config.SUBMISSION_FILE, help="Path to output CSV")
    args = parser.parse_args()

    pipeline_start = time.time()

    # 0. Setup
    Config.seed_everything()
    Config.ensure_dirs()
    logger.info("=" * 70)
    logger.info("RedRob Candidate Ranking Pipeline — Starting")
    logger.info("=" * 70)

    # 1. Validate JD exists (CHANGE 1 — no hardcoded JD)
    jd_path = args.jd
    if not os.path.exists(jd_path):
        logger.error(
            f"Job description not found at {jd_path}. "
            f"Please place your job_description.md in the data/ directory."
        )
        sys.exit(1)

    # 2. Parse Job Description
    logger.info("Phase 1: Parsing job description...")
    jd_features = parse_job_description(jd_path)
    jd_raw_text = open(jd_path, "r", encoding="utf-8").read()
    logger.info(f"  Must-have skills: {jd_features.must_have_skills[:10]}")
    logger.info(f"  Experience range: {jd_features.experience_range}")
    logger.info(f"  Domain keywords: {jd_features.domain_keywords[:10]}")

    # 3. Load, Validate, Pre-filter, Parse, and Extract Structured Features using a Streaming Heap
    logger.info("Phase 2+3+4: Streaming candidates, pre-filtering, and calculating structured features...")
    t0 = time.time()

    min_yoe = Config.EXPERIENCE_IDEAL_MIN - 2  # 5 - 2 = 3

    # Heap will contain tuples of (structured_score, -idx, candidate_id, candidate, features)
    # Using -idx as a tie-breaker ensures that if there are identical structured scores,
    # the candidate appearing earlier in the file (smaller idx) will be preferred, matching stable sort descending.
    heap = []
    invalid_count = 0
    prefilter_removed = 0
    processed_count = 0

    filepath = Config.CANDIDATES_FILE

    for idx, record in enumerate(stream_jsonl(filepath)):
        if args.limit is not None and idx >= args.limit:
            break

        is_valid, reason = validate_record(record)
        if not is_valid:
            invalid_count += 1
            continue

        # Cheap pre-filter on raw dict BEFORE parsing
        if not fast_prefilter_raw(record, min_yoe=min_yoe):
            prefilter_removed += 1
            continue

        candidate = parse_candidate(record)
        features = build_structured_features(candidate, jd_features)
        structured_score = features["structured_score"]

        # Keep only the top-K by structured score using a heap
        if len(heap) < Config.STRUCTURED_TOP_K:
            heapq.heappush(heap, (structured_score, -idx, candidate.candidate_id, candidate, features))
        else:
            # If the current score is higher than the root of our min-heap, push it and pop the root.
            if structured_score > heap[0][0]:
                heapq.heappushpop(heap, (structured_score, -idx, candidate.candidate_id, candidate, features))

        processed_count += 1

    # Reconstruct top_k_candidates and candidate_features preserving the original file order
    # Sort the heap by original file index (idx) to preserve order
    heap.sort(key=lambda x: -x[1])

    top_k_candidates: List[Candidate] = []
    candidate_features: Dict[str, Dict[str, Any]] = {}
    top_k_ids = set()

    for structured_score, neg_idx, cid, candidate, features in heap:
        top_k_candidates.append(candidate)
        candidate_features[cid] = features
        top_k_ids.add(cid)

    logger.info(
        f"  Processed: {processed_count} candidates, "
        f"{invalid_count} invalid, {prefilter_removed} pre-filtered "
        f"in {time.time() - t0:.1f}s"
    )
    logger.info(f"  Selected top {len(top_k_candidates)} candidates by structured score for semantic reranking")

    # 7. Semantic Embedding & Similarity
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

    # Concept embeddings (TASK A)
    logger.info("  Computing concept embeddings...")
    concept_embeddings = store.load_concept_embeddings()
    if concept_embeddings is None:
        if embedder.model is None:
            embedder.load_model()
        concept_texts = {
            "ranking": RANKING_CONCEPT_TEXT,
            "evaluation": EVALUATION_CONCEPT_TEXT,
            "production": PRODUCTION_CONCEPT_TEXT,
        }
        concept_embeddings = embedder.encode_concept_texts(concept_texts)
        store.save_concept_embeddings(concept_embeddings)
        logger.info("  Concept embeddings computed and cached")
    else:
        logger.info("  Using cached concept embeddings")

    # Compute concept scores for all top-k candidates
    concept_scores = compute_concept_scores(
        concept_embeddings, candidate_embeddings, candidate_ids_list
    )

    logger.info(f"  Semantic + concept scoring complete in {time.time() - t0:.1f}s")

    # 8. Score Combination (with concept scores)
    logger.info("Phase 7: Score combination...")

    # Only combine scores for candidates that went through semantic
    top_k_features = {cid: candidate_features[cid] for cid in top_k_ids}
    # Use default quality score for now (detailed honeypot runs post-scoring)
    top_k_quality = {cid: 0.5 for cid in top_k_ids}

    scored_candidates = combine_scores(
        top_k_features, semantic_scores, top_k_quality,
        concept_scores=concept_scores,
    )

    # 9. Detailed Honeypot on Top 300 (TASK E — staged)
    logger.info("Phase 8: Detailed honeypot on top 300 post-scoring...")
    t0 = time.time()

    # Sort by final_score, take top 300
    scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
    top_300 = scored_candidates[:300]
    candidate_map_all = {c.candidate_id: c for c in top_k_candidates}

    # Run detailed quality scoring on top 300
    quality_survivors = []
    for entry in top_300:
        cid = entry["candidate_id"]
        candidate = candidate_map_all.get(cid)
        if candidate:
            q_score = compute_quality_score(candidate)
            candidate.quality_score = q_score
            entry["quality_score"] = q_score
            if q_score >= Config.QUALITY_FILTER_THRESHOLD:
                quality_survivors.append(entry)
        else:
            quality_survivors.append(entry)

    # If fewer than 100 survive from top 300, add more from remaining
    if len(quality_survivors) < Config.TOP_N:
        remaining = scored_candidates[300:]
        for entry in remaining:
            if len(quality_survivors) >= Config.TOP_N:
                break
            entry["quality_score"] = 0.5  # default for non-honeypot-checked
            quality_survivors.append(entry)

    removed_honeypot = len(top_300) - len([e for e in quality_survivors if e in top_300])
    logger.info(
        f"  Detailed honeypot: {len(quality_survivors)} survived from top 300, "
        f"in {time.time() - t0:.1f}s"
    )

    # 10. Rank & Select Top 100
    logger.info("Phase 9: Ranking and selection...")
    ranked = rank_candidates(quality_survivors, top_n=Config.TOP_N)

    # 11. Reasoning Generation
    logger.info("Phase 10: Generating reasoning...")
    candidate_map = {c.candidate_id: c for c in top_k_candidates}
    ranked = generate_reasoning(ranked, candidate_map, jd_features)

    # 12. Build & Validate Submission
    logger.info("Phase 11: Building submission CSV...")
    df = build_submission(ranked, output_path=args.output)

    # Generate debug CSV with original details
    build_debug_csv(ranked, candidate_map)

    # Also save full scores for analysis
    save_full_scores(scored_candidates)

    # Done
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
