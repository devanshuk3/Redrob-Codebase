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
from src.ingestion.schema_validator import validate_record
from src.ingestion.candidate_parser import Candidate, parse_candidate
from src.ingestion.honeypot_filter import (
    compute_quality_score,
    _check_non_tech_role,
    fast_prefilter_raw,
    is_hard_excluded,
    get_candidate_issues,
    detect_lsh_clusters,       
)
from src.features.jd_feature_mapper import parse_job_description, JDFeatures
from src.features.feature_builder import build_structured_features
from src.semantic.text_builder import build_candidate_text, build_jd_text
from src.semantic.embedding_generator import EmbeddingGenerator
from src.semantic.embedding_store import EmbeddingStore
from src.semantic.semantic_ranker import compute_semantic_scores, compute_concept_scores
from src.semantic.cross_encoder_reranker import compute_cross_encoder_scores
from src.ranking.score_combiner import combine_scores
from src.ranking.ranker import rank_candidates
from src.ranking.mmr_reranker import mmr_rerank
from src.ranking.reasoning_generator import generate_reasoning
from src.ranking.submission_builder import (
    build_submission,
    save_full_scores,
    build_debug_csv,
    build_honeypot_debug_csv, 
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.constants import (
    RANKING_CONCEPT_TEXT,
    EVALUATION_CONCEPT_TEXT,
    PRODUCTION_CONCEPT_TEXT,
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

    # Count total candidates in the input candidates file to determine if we have less than 100
    total_candidates = 0
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total_candidates += 1
    effective_candidates = total_candidates
    if args.limit is not None:
        effective_candidates = min(total_candidates, args.limit)

    # Dynamic TOP_N adjustment if candidate file is less than 100
    if effective_candidates < 100:
        Config.TOP_N = effective_candidates
        logger.info(f"Input has less than 100 candidates ({effective_candidates}). Updating Config.TOP_N to {Config.TOP_N}.")
    else:
        Config.TOP_N = 100

    prefiltered_records = []

    for idx, record in enumerate(stream_jsonl(filepath)):
        if args.limit is not None and idx >= args.limit:
            break

        is_valid, reason = validate_record(record)
        if not is_valid:
            invalid_count += 1
            continue

        # Cheap pre-filter on raw dict BEFORE parsing
        if effective_candidates >= 100 and not fast_prefilter_raw(record, min_yoe=min_yoe):
            prefilter_removed += 1
            prefiltered_records.append((idx, record))
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

    # Fallback: if we have fewer than Config.TOP_N candidates processed, pull from prefiltered_records
    if len(heap) < Config.TOP_N and prefiltered_records:
        for idx, record in prefiltered_records:
            if len(heap) >= Config.TOP_N:
                break
            candidate = parse_candidate(record)
            features = build_structured_features(candidate, jd_features)
            structured_score = features["structured_score"]
            heapq.heappush(heap, (structured_score, -idx, candidate.candidate_id, candidate, features))
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

    # ---- HYBRID: Lexical (TF-IDF) scoring ----
    logger.info("  Computing lexical (TF-IDF) scores for hybrid retrieval...")
    t_lex = time.time()
    
    lexical_scores_dict = {}
    if candidate_texts:
        # Prepare texts (already built earlier in the pipeline)
        # jd_text and candidate_texts are already defined above
        all_texts = [jd_text] + candidate_texts
        vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        jd_vec = tfidf_matrix[0:1]
        candidate_vecs = tfidf_matrix[1:]
        
        lexical_similarities = cosine_similarity(jd_vec, candidate_vecs).flatten()
        lexical_scores = np.clip(lexical_similarities, 0.0, 1.0)
        
        lexical_scores_dict = {
            cid: float(score) for cid, score in zip(candidate_ids_list, lexical_scores)
        }
    
    logger.info(f"  Lexical scoring complete in {time.time() - t_lex:.1f}s")

    # 8. Score Combination (with concept scores)
    logger.info("Phase 7: Score combination...")

    # Only combine scores for candidates that went through semantic
    top_k_features = {cid: candidate_features[cid] for cid in top_k_ids}
    # Use default quality score for now (detailed honeypot runs post-scoring)
    top_k_quality = {cid: 0.5 for cid in top_k_ids}
  
    scored_candidates = combine_scores(
        top_k_features, semantic_scores, top_k_quality,
        concept_scores=concept_scores,lexical_scores=lexical_scores_dict,
    )

    # 9. Detailed Honeypot on Top 300 (TASK E — staged)
        
    logger.info("Phase 8: Detailed honeypot on top 300 post-scoring...")
    t0 = time.time()

    # Sort by final_score, take top 300
    scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
    top_300 = scored_candidates[:300]
    candidate_map_all = {c.candidate_id: c for c in top_k_candidates}
    
    # 9a. Cross-Encoder Reranking on Top 300 (joint JD+candidate relevance)
    logger.info("Phase 8a: Cross-encoder reranking on top 300...")
    t0_ce = time.time()

    cross_encoder_candidates = [
        candidate_map_all[e["candidate_id"]]
        for e in top_300
        if e["candidate_id"] in candidate_map_all
    ]
    cross_encoder_scores = compute_cross_encoder_scores(jd_raw_text, cross_encoder_candidates)

    # Normalize within this batch before using it as a modifier. Cross-encoder
    # models trained on short web-search query/passage pairs (like
    # ms-marco-MiniLM) are often badly miscalibrated on long-form JD-vs-profile
    # text — raw sigmoid outputs can cluster far from 0.5 (e.g. all under 0.12)
    # even when the model IS meaningfully differentiating between candidates.
    # Anchoring the modifier to a fixed 0.5 midpoint in that case applies an
    # almost uniform shrink to everyone instead of an actual rerank. Min-max
    # normalizing within the batch uses the model's real relative ordering
    # instead of assuming its absolute scale means anything here.
    ce_values = list(cross_encoder_scores.values())
    ce_min, ce_max = (min(ce_values), max(ce_values)) if ce_values else (0.0, 1.0)
    ce_range = ce_max - ce_min

    for entry in top_300:
        ce_raw = cross_encoder_scores.get(entry["candidate_id"], 0.5)
        entry["cross_encoder_score"] = ce_raw
        ce_norm = (ce_raw - ce_min) / ce_range if ce_range > 1e-9 else 0.5
        entry["cross_encoder_score_normalized"] = round(ce_norm, 4)
        # Placeholder multiplicative modifier — once cross_encoder_score is
        # wired into score_combiner.py as a learned-fusion feature, this
        # block can go away entirely.
        ce_mult = 1.0 + Config.CROSS_ENCODER_MODIFIER_STRENGTH * (ce_norm - 0.5)
        entry["final_score"] = round(entry["final_score"] * ce_mult, 6)

    top_300.sort(key=lambda x: x["final_score"], reverse=True)
    logger.info(f"  Cross-encoder reranking complete in {time.time() - t0_ce:.1f}s")

    
    # --------------------------------------------------------------
    # NEW: MinHash-LSH cluster detection on Top 300 (Scalable)
    # --------------------------------------------------------------
   
    cluster_honeypot_ids = detect_lsh_clusters(top_300, candidate_map_all, threshold=0.85, min_cluster_size=3)
    # --------------------------------------------------------------

    # Run detailed quality scoring on top 300 (skip clustered profiles)
    quality_survivors = []
    hard_excluded_count = 0
    cluster_skipped_count = 0
    quality_filtered_count = 0
    removed_honeypots = []

    for entry in top_300:
        cid = entry["candidate_id"]
        candidate = candidate_map_all.get(cid)
        reason = None
        details = ""
        issue_names = ""

        # If effective_candidates < 100, we do not filter out anyone
        if effective_candidates < 100:
            q_score = compute_quality_score(candidate) if candidate else 0.5
            if candidate:
                candidate.quality_score = q_score
            entry["quality_score"] = q_score
            
            semantic = entry["semantic_score"]
            structured = entry["structured_score"]
            behavioral = entry["behavioral_score"]
            base_initial = (
                0.20 * semantic
                + 0.55 * structured
                + 0.15 * behavioral
                + 0.10 * 0.5
            )
            base_new = (
                0.20 * semantic
                + 0.55 * structured
                + 0.15 * behavioral
                + 0.10 * q_score
            )
            if base_initial > 0:
                entry["final_score"] = round(entry["final_score"] * (base_new / base_initial), 6)
            else:
                entry["final_score"] = 0.0
            
            quality_survivors.append(entry)
            continue

        # --- Check 1: Cluster detection ---
        if cid in cluster_honeypot_ids:
            cluster_skipped_count += 1
            reason = "clustered"
            details = "Near-identical text profile (>90% similarity) to other candidates in top 300."
            issue_names = "CLUSTERED_TEMPLATE"
            removed_honeypots.append({
                "entry": entry,
                "candidate": candidate,
                "reason": reason,
                "details": details,
                "issue_names": issue_names
            })
            continue

        if not candidate:
            quality_survivors.append(entry)
            continue

        # --- Check 2: Hard exclusion ---
        if is_hard_excluded(candidate):
            hard_excluded_count += 1
            reason = "hard_excluded"
            issues = get_candidate_issues(candidate)
            details = ", ".join(issues.keys())
            issue_names = ", ".join(issues.keys())
            removed_honeypots.append({
                "entry": entry,
                "candidate": candidate,
                "reason": reason,
                "details": details,
                "issue_names": issue_names
            })
            continue

        # --- Check 3: Quality score ---
        q_score = compute_quality_score(candidate)
        candidate.quality_score = q_score
        entry["quality_score"] = q_score
        
        semantic = entry["semantic_score"]
        structured = entry["structured_score"]
        behavioral = entry["behavioral_score"]
        base_initial = (
            0.20 * semantic
            + 0.55 * structured
            + 0.15 * behavioral
            + 0.10 * 0.5
        )
        base_new = (
            0.20 * semantic
            + 0.55 * structured
            + 0.15 * behavioral
            + 0.10 * q_score
        )
        if base_initial > 0:
            entry["final_score"] = round(entry["final_score"] * (base_new / base_initial), 6)
        else:
            entry["final_score"] = 0.0
            
        if q_score >= Config.QUALITY_FILTER_THRESHOLD:
            quality_survivors.append(entry)
        else:
            quality_filtered_count += 1
            reason = "quality_filtered"
            details = f"q_score={q_score:.4f} < threshold {Config.QUALITY_FILTER_THRESHOLD}"
            issue_names = "QUALITY_FILTERED"
            removed_honeypots.append({
                "entry": entry,
                "candidate": candidate,
                "reason": reason,
                "details": details,
                "issue_names": issue_names
            })

    # If fewer than Config.TOP_N survive from top 300, add more from remaining
    if len(quality_survivors) < Config.TOP_N:
        remaining = scored_candidates[300:]
        for entry in remaining:
            if len(quality_survivors) >= Config.TOP_N:
                break
            entry["quality_score"] = 0.5
            quality_survivors.append(entry)

    # If still fewer than Config.TOP_N survive, add back from removed_honeypots
    if len(quality_survivors) < Config.TOP_N and removed_honeypots:
        for item in removed_honeypots:
            if len(quality_survivors) >= Config.TOP_N:
                break
            entry = item["entry"]
            quality_survivors.append(entry)

    # Dynamic TOP_N adjustment if fewer than Config.TOP_N survive
    if len(quality_survivors) < Config.TOP_N:
        logger.info(f"Fewer than {Config.TOP_N} candidates survived filtering. Updating Config.TOP_N to {len(quality_survivors)}.")
        Config.TOP_N = len(quality_survivors)

    removed_honeypot = len(top_300) - len([e for e in quality_survivors if e in top_300])
    quality_filtered = removed_honeypot - hard_excluded_count - cluster_skipped_count
    logger.info(
        f"  Detailed honeypot: {len(quality_survivors)} survived from top 300 "
        f"({hard_excluded_count} hard-excluded, {cluster_skipped_count} clustered, {quality_filtered_count} quality-filtered), "
        f"in {time.time() - t0:.1f}s"
    )
        # --------------------------------------------------------------
    # Save debug CSV for all removed honeypots
    # --------------------------------------------------------------
    build_honeypot_debug_csv(removed_honeypots)

    # 9.5. MMR Diversity Reranking (optional)
    quality_survivors = mmr_rerank(
        quality_survivors, candidate_embeddings, candidate_ids_list,
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
    if ranked:
        logger.info(f"Top candidate: {ranked[0]['candidate_id']} "
                    f"(score: {ranked[0]['final_score']:.4f})")
        logger.info(f"Last ({len(ranked)}th) candidate: {ranked[-1]['candidate_id']} "
                    f"(score: {ranked[-1]['final_score']:.4f})")
    logger.info("=" * 70)

    return df


if __name__ == "__main__":
    main()
