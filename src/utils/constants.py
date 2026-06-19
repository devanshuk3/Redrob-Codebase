"""
Keyword constants for feature extraction.
Organized by detection domain — all lowercase for matching.
"""

# Target Skills (detected via fuzzy matching against JD)
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

# Retrieval Domain Keywords (CHANGE 2 — strengthened)
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
    # Strengthened: additional retrieval signals
    "search relevance", "query expansion", "search ranking",
    "candidate retrieval", "search infrastructure", "search system",
    "retrieval system", "embedding pipeline", "embedding index",
    "vector index", "nearest neighbor", "knn", "hnsw",
    "qdrant", "index refresh", "inverted index", "search quality",
]

# Retrieval skill keywords for bonus scoring
RETRIEVAL_SKILL_KEYWORDS = {
    "retrieval", "search", "rag", "vector", "elasticsearch",
    "milvus", "pinecone", "weaviate", "faiss", "solr", "lucene",
    "opensearch", "qdrant", "bm25", "information retrieval",
    "semantic search", "hybrid search", "vector database",
    "vector search", "dense retrieval", "neural search",
}

# Ranking/Recommendation Keywords (CHANGE 2 — strengthened)
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
    # Strengthened: additional ranking signals
    "learning-to-rank", "l2r", "candidate ranking", "result ranking",
    "re-ranking", "ranking pipeline", "ranking quality",
    "recommendation engine", "recommendation pipeline",
    "candidate matching", "job matching", "talent matching",
]

# Ranking skill keywords for bonus scoring
RANKING_SKILL_KEYWORDS = {
    "ranking", "recommendation", "recommender", "matching",
    "personalization", "collaborative filtering", "xgboost",
    "learning to rank", "lambdamart", "lightgbm",
    "candidate matching", "search ranking",
}

# Evaluation Framework Keywords (CHANGE 2 — strengthened)
EVALUATION_KEYWORDS = [
    "ndcg", "map", "mrr", "recall@k", "precision@k",
    "a/b testing", "online evaluation", "offline evaluation",
    "evaluation framework", "metrics pipeline", "model evaluation",
    "statistical significance", "interleaving", "holdout testing",
    "backtesting", "counterfactual evaluation", "causal inference",
    "experiment platform", "feature importance", "ablation study",
    "cross validation", "hyperparameter tuning",
    # Strengthened: additional evaluation signals
    "recall@", "precision@", "f1 score", "roc auc",
    "mean average precision", "mean reciprocal rank",
    "normalized discounted cumulative gain",
    "experiment framework", "online experiment",
    "offline metrics", "ranking metrics", "search metrics",
    "evaluation pipeline", "evaluation metrics", "metric improvement",
    "benchmark", "test set", "ground truth",
]

# Production/Deployment Keywords (CHANGE 3 — strengthened)
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
    # Strengthened: additional production signals
    "inference serving", "model deployment", "production system",
    "production environment", "production infrastructure",
    "qps", "queries per second", "latency optimization",
    "index refresh", "real-time", "low latency",
    "scalable", "scalability", "horizontal scaling",
    "production ml", "production machine learning",
    "inference optimization", "model optimization",
    "system design", "architecture design",
]

# LLM Hype Keywords (TASK 3)
LLM_HYPE_KEYWORDS = [
    "langchain", "openai", "prompt engineering", "chatgpt",
    "claude", "gemini", "gpt-4", "gpt-3", "gpt3", "gpt4",
    "llamaindex", "llama index", "prompt tuning", "prompt design",
    "chatbot", "conversational ai", "ai assistant",
    "openai api", "anthropic", "copilot",
]

# Keywords that indicate real ML depth (counterbalance to LLM hype)
REAL_ML_DEPTH_KEYWORDS = [
    "retrieval", "ranking", "recommendation", "search",
    "evaluation", "production ml", "deployed", "serving",
    "embeddings", "fine-tuning", "training", "inference",
    "distributed", "pipeline", "scale", "latency",
    "vector database", "information retrieval",
    "machine learning", "deep learning", "model training",
    "feature engineering", "data pipeline",
]

# Education: Relevant Fields
RELEVANT_EDUCATION_FIELDS = [
    "computer science", "artificial intelligence", "machine learning",
    "data science", "software engineering", "information technology",
    "computer engineering", "electrical engineering", "mathematics",
    "statistics", "computational linguistics", "cognitive science",
]

# Degree Level Weights
DEGREE_WEIGHTS = {
    "phd": 1.0, "ph.d": 1.0, "doctorate": 1.0,
    "masters": 0.8, "master": 0.8, "m.s.": 0.8, "m.sc": 0.8,
    "m.tech": 0.85, "m.e.": 0.8, "mba": 0.6,
    "bachelors": 0.6, "bachelor": 0.6, "b.s.": 0.6, "b.sc": 0.6,
    "b.tech": 0.65, "b.e.": 0.6,
    "diploma": 0.3, "associate": 0.3,
}

# Certification Keywords
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

# Tier Weights (institution quality)
TIER_WEIGHTS = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.5,
    "tier_4": 0.25,
}

# Proficiency Multipliers
PROFICIENCY_WEIGHTS = {
    "expert": 1.0,
    "advanced": 1.0,
    "intermediate": 0.7,
    "beginner": 0.4,
}

# Skill Domains for Consistency Checking (TASK 7)
UNRELATED_SKILL_DOMAINS = {
    "mechanical": ["mechanical design", "autocad", "solidworks", "catia", "ansys",
                    "mechanical engineering", "thermodynamics", "fluid mechanics"],
    "civil": ["civil engineering", "structural analysis", "construction",
              "surveying", "geotechnical"],
    "accounting": ["accounting", "bookkeeping", "tax", "audit", "financial reporting",
                    "tally", "quickbooks"],
    "sales": ["sales", "cold calling", "lead generation", "crm", "salesforce",
              "business development"],
    "design": ["photoshop", "illustrator", "figma", "sketch", "indesign",
               "graphic design", "ui design", "visual design"],
    "hr": ["recruitment", "talent acquisition", "payroll", "onboarding",
           "employee relations", "hr management"],
}

# Consulting/IT-services companies — single source of truth (Fix #2)
# JD explicitly says: "People who have only worked at consulting firms ... in their entire career"
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "tata consultancy services",
    "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "hcl technologies", "hcltech",
    "tech mahindra",
    "mindtree", "ltimindtree", "lti", "l&t infotech", "larsen & toubro infotech",
    "mphasis",
    "hexaware", "hexaware technologies",
    "cyient",
    "zensar", "zensar technologies",
    "niit technologies", "coforge",
    "persistent systems", "persistent",
}

# Concept Embedding Texts (TASK A — 3-concept semantic scoring)

RANKING_CONCEPT_TEXT = """
Built and deployed production ranking or retrieval systems. Dense retrieval,
semantic search, vector databases, FAISS, Pinecone, Weaviate, Qdrant, Milvus,
Elasticsearch, OpenSearch. Recommendation systems, information retrieval,
learning to rank, re-ranking, BM25, hybrid search. Sentence transformers,
bi-encoder models, cross-encoders, embedding models, RAG systems.
"""

EVALUATION_CONCEPT_TEXT = """
Designed evaluation frameworks for ranking and retrieval systems. Offline
evaluation metrics NDCG, MRR, MAP, Precision at K. A/B testing for ranking
quality. Online to offline correlation. Statistical significance testing.
Recall evaluation, precision-recall tradeoffs, benchmark datasets.
"""

PRODUCTION_CONCEPT_TEXT = """
Shipped ML systems to production at scale. Fine-tuning LLMs with LoRA, QLoRA,
PEFT, instruction tuning. Strong Python engineering, production-grade code.
MLOps, experiment tracking, MLflow, Weights and Biases. Model serving,
inference optimization. Open source contributions in AI and ML.
"""

