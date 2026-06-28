# Intelligent Candidate Discovery & Ranking System

An enterprise-grade, CPU-optimized hybrid scoring and ranking pipeline designed for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**.

This system processes candidate profiles against a job description, filters out synthetic/honeypot profiles, scores candidates across multiple dimensions, and selects the **Top 100 best-suited candidates** with automated reasoning.

---

## 🚀 Key Features

* **Hybrid Scoring**: Combines Semantic Relevance (SentenceTransformers), Structured Features (Fuzzy Match, YOE Fit), and Behavioral Signals.
* **Honeypot Filter**: Multi-stage detection of unrealistic skill counts and invalid timelines.
* **MMR Reranking**: Optional diversity-promoting step using Maximal Marginal Relevance.
* **Fully Local**: 100% offline inference using CPU-optimized batching.

---

## 🛠️ Pipeline Flow

The system runs in the following sequence:

```
[Raw Candidates] ──> [Schema Validation & Streaming Loader]
                               │
                               ▼
         [Fast Pre-Filtering] (Exclude non-tech roles & low YOE)
                               │
                               ▼
                 [Structured Feature Scoring]
                               │
                               ▼
    [Streaming Min-Heap Selection] (Select top 5,000 structured candidates)
                               │
                               ▼
             [Semantic Scoring] (SentenceTransformers)
                               │
                               ▼
                     [Weighted Score Fusion]
                               │
                               ▼
    [Detailed Honeypot Filtering] (Evaluated on top 300 candidates)
                               │
                               ▼
     [MMR Diversity Reranking] (Optional - top 150 reranked to 100)
                               │
                               ▼
                 [Ranker, Tie-Breaker & Reasoning]
                               │
                               ▼
                   [Outputs / Submission CSV]
```

---

## ⚙️ MMR Diversity Reranking

A lightweight Maximal Marginal Relevance (MMR) step is inserted immediately before the final top-100 selection to balance score relevance and profile diversity.

### Configuration

You can enable or customize MMR in `src/ranking/mmr_reranker.py`:

```python
ENABLE_MMR = False   # Set to True to enable MMR. When False, output matches original system.
MMR_LAMBDA = 0.85    # Trade-off parameter (higher = relevance, lower = diversity)
MMR_POOL_SIZE = 150  # Candidates pulled from the top of the scored pool
MMR_SELECT_N = 100   # Number of final candidates to select
```

---

## 💻 Setup & Execution

### 1. Prerequisites
* **Python**: `3.10` or higher
* **RAM**: 8 GB minimum (16 GB recommended)

### 2. Installation
Navigate to the root directory and install dependencies:

```bash
# 1. Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Pipeline
Ensure input files are placed in `data/candidates.jsonl` and `data/job_description.md`, then run:

```bash
# Run on the full dataset
python main.py

# Run on a smaller subset (e.g. 5000 candidates) for testing
python main.py --limit 5000

# Specify custom inputs/outputs
python main.py --jd data/job_description.md --output outputs/submission.csv
```

### 4. Run Verification & Tests

```bash
# Validate the submission output format
python "../Descriptions and protocol/validate_submission.py" outputs/submission.csv

# Run the test suite
pytest -v
```
