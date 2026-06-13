"""
Keyword constants for feature extraction.
Organized by detection domain — all lowercase for matching.
"""

# ── Target Skills (detected via fuzzy matching against JD) ────────────
# These are fallback defaults; the JD feature mapper will override them.
DEFAULT_TARGET_SKILLS = [
    "python", "machine learning", "deep learning", "pytorch", "tensorflow",
    "embeddings", "vector databases", "retrieval systems", "search systems",
    "ranking systems", "recommendation systems", "llms", "fine tuning",
    "lora", "qlora", "peft", "rag", "mlops", "docker", "kubernetes",
    "langchain", "llamaindex", "huggingface", "transformers", "nlp",
    "computer vision", "data pipelines", "sql", "nosql", "redis",
    "elasticsearch", "milvus", "pinecone", "weaviate", "faiss",
]

# ── Retrieval Domain Keywords (CHANGE 2) ─────────────────────────────
RETRIEVAL_KEYWORDS = [
    "semantic search", "retrieval", "hybrid search", "vector search",
    "rag", "retrieval augmented generation", "candidate matching",
    "document retrieval", "information retrieval", "search engine",
    "dense retrieval", "sparse retrieval", "bm25", "tf-idf",
    "neural search", "passage retrieval", "query understanding",
    "reranking", "cross encoder", "bi-encoder", "colbert",
    "embedding search", "approximate nearest neighbor", "ann",
    "faiss", "milvus", "pinecone", "weaviate", "elasticsearch",
    "opensearch", "solr", "lucene",
]

# ── Ranking/Recommendation Keywords (CHANGE 2) ──────────────────────
RANKING_KEYWORDS = [
    "ranking systems", "recommendation systems", "recommender systems",
    "matching systems", "learning to rank", "search ranking",
    "candidate ranking", "content recommendation", "collaborative filtering",
    "matrix factorization", "click-through rate", "ctr prediction",
    "personalization", "item recommendation", "user modeling",
    "two-tower model", "multi-stage ranking", "feature store",
    "ranking model", "relevance scoring", "listwise ranking",
    "pairwise ranking", "pointwise ranking", "xgboost ranking",
    "lambdamart", "ndcg optimization",
]

# ── Evaluation Framework Keywords (CHANGE 2) ────────────────────────
EVALUATION_KEYWORDS = [
    "ndcg", "map", "mrr", "recall@k", "precision@k",
    "a/b testing", "online evaluation", "offline evaluation",
    "evaluation framework", "metrics pipeline", "model evaluation",
    "statistical significance", "interleaving", "holdout testing",
    "backtesting", "counterfactual evaluation", "causal inference",
    "experiment platform", "feature importance", "ablation study",
    "cross validation", "hyperparameter tuning",
]

# ── Production/Deployment Keywords (CHANGE 3) ───────────────────────
PRODUCTION_KEYWORDS = [
    "deployed", "production", "latency", "serving", "monitoring",
    "observability", "millions of users", "scale", "distributed systems",
    "reliability", "a/b testing", "mlops", "ci/cd", "model serving",
    "real-time inference", "batch inference", "feature engineering",
    "data pipeline", "etl", "airflow", "kubeflow", "mlflow",
    "model registry", "canary deployment", "blue-green deployment",
    "load balancing", "auto scaling", "sla", "slo",
    "kubernetes", "docker", "microservices", "api", "rest api",
    "grpc", "kafka", "rabbitmq", "redis", "caching",
    "high availability", "fault tolerance", "disaster recovery",
]

# ── Education: Relevant Fields ───────────────────────────────────────
RELEVANT_EDUCATION_FIELDS = [
    "computer science", "artificial intelligence", "machine learning",
    "data science", "software engineering", "information technology",
    "computer engineering", "electrical engineering", "mathematics",
    "statistics", "computational linguistics", "cognitive science",
]

# ── Degree Level Weights ─────────────────────────────────────────────
DEGREE_WEIGHTS = {
    "phd": 1.0, "ph.d": 1.0, "doctorate": 1.0,
    "masters": 0.8, "master": 0.8, "m.s.": 0.8, "m.sc": 0.8,
    "m.tech": 0.85, "m.e.": 0.8, "mba": 0.6,
    "bachelors": 0.6, "bachelor": 0.6, "b.s.": 0.6, "b.sc": 0.6,
    "b.tech": 0.65, "b.e.": 0.6,
    "diploma": 0.3, "associate": 0.3,
}

# ── Certification Keywords ───────────────────────────────────────────
RELEVANT_CERTIFICATIONS = [
    "aws machine learning", "aws ml", "aws certified machine learning",
    "google cloud machine learning", "google ml", "gcp ml",
    "azure ai", "azure machine learning",
    "databricks", "databricks certified",
    "deeplearning.ai", "deep learning specialization",
    "nvidia", "nvidia deep learning",
    "mlops", "ml engineering",
    "tensorflow developer", "pytorch",
    "huggingface", "langchain",
]

# ── Tier Weights (institution quality) ───────────────────────────────
TIER_WEIGHTS = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.5,
    "tier_4": 0.25,
}

# ── Proficiency Multipliers ──────────────────────────────────────────
PROFICIENCY_WEIGHTS = {
    "expert": 1.0,
    "advanced": 0.85,
    "intermediate": 0.6,
    "beginner": 0.3,
}
