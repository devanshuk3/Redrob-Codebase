# Intelligent Candidate Discovery & Ranking System

An enterprise-grade, CPU-optimized hybrid scoring and ranking pipeline designed for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**.

This system processes candidate profiles against a job description, filters out synthetic/honeypot profiles, scores candidates across multiple dimensions, and selects the **Top 100 best-suited candidates** with automated reasoning.

---

## Pipeline Architecture

Below is the visual overview of the different processing phases in the pipeline:

![System Pipeline Phases](phases.png)

---

## Key Features

* **Schema Validation & Streaming Loader**: Memory-efficient streaming of raw candidate profiles with real-time JSON schema validation.
* **Hybrid Scoring Engine**: Combines Semantic Relevance (SentenceTransformers), Structured Features (Fuzzy Match, YOE Fit), and Behavioral Signals.
* **Advanced Honeypot Detection**: 
  - **Fast pre-filtering**: Early exclusion of non-tech roles and candidates below minimum YOE.
  - **MinHash-LSH Clustering**: Grouping and skipping template-generated near-identical profiles.
  - **Quality Scoring**: Multi-stage validation of unrealistic skill counts, invalid timelines, and chronological inconsistencies.
* **Cross-Encoder Reranking**: Utilizes a cross-encoder model on the top candidate pool to evaluate joint relevance with the job description.
* **MMR Reranking**: Optional diversity-promoting step using Maximal Marginal Relevance.
* **Explainable AI**: Generates automated, human-readable structured reasoning for every ranked candidate.
* **Fully Local & CPU-Optimized**: 100% offline inference with CPU-optimized PyTorch execution for maximum resource efficiency.

---

## Setup & Execution

### 1. Cold Start (Recommended & Automated)
For a completely automated setup, place `candidates.jsonl` and `job_description.md` in the project root and run:
```bash
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
This script will automatically:
1. Create a virtual environment (`venv`) if it doesn't already exist.
2. Upgrade `pip`.
3. Install all packages or dependencies.
4. Run the full candidate ranking pipeline.

---

### 2. Manual Setup (Alternative)
If you prefer setting up the environment manually:

```bash
# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip & install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Running the Pipeline via CLI
Place `candidates.jsonl` and `job_description.md` in your working directory, then run:

```bash
# Run the pipeline (defaults: ./candidates.jsonl, ./job_description.md, ./submission.csv)
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Run on a smaller subset (e.g. 5,000 candidates) for rapid testing
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv --limit 5000

# Specify a custom job description
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv --jd ./job_description.md
```

---

### 4. Running the Sandbox App
To start the Streamlit web dashboard manually:
```bash
streamlit run sandbox/app.py
```

---

### 5. Running Tests
To run the test suite:
```bash
pytest -v
```

---

##  Output Artifacts
All generated files are written relative to the `--out` path:
* **`./submission.csv`**: The final Top 100 ranked candidates with IDs and structured reasoning (matching submission schema).
* **`./candidate_scores.csv`**: Comprehensive breakdown of scores (semantic, structured, quality, lexical, etc.) for all candidates.
* **`./debug/candidate_debug.csv`**: Detailed debug CSV with full score breakdowns for ranked candidates.
* **`./debug/honeypots_detailed.csv`**: Detailed reasons/issues for profiles flagged and removed by the honeypot detection filters.
* **`./logs/pipeline.log`**: Full pipeline execution log.
