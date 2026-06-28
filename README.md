# Intelligent Candidate Discovery & Ranking System

An enterprise-grade, CPU-optimized hybrid scoring and ranking pipeline designed for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**. 

This system processes **100,000 candidate profiles** against a given job description, automatically detects and filters out synthetic/honeypot profiles, scores candidates across multiple dimensions (semantic relevance, structured features, behavioral signals, and profile quality), and returns the **Top 100 best-suited candidates** with detailed reasoning.

---

## Project Phases & Workflow

The system is developed and orchestrated following a structured 11-step pipeline designed for maximum speed, memory efficiency, and scoring accuracy.

```
[Raw Candidates (100k)] ──> [Schema Validation & Streaming Loader]
                                       │
                                       ▼
                              [Fast Pre-Filtering] (Non-tech roles & YOE minimum excluded)
                                       │
                                       ▼
                         [Structured Feature Scoring]
                         - JD Skill Fuzzy Matching (RapidFuzz)
                         - Experience & Education Fit Scoring
                         - Behavioral Signals Mapping (Notice period, reloc, etc.)
                                       │
                                       ▼
                       [Streaming Min-Heap Selection] (Retrieves Top 5,000 candidates by structured score)
                                       │
                                       ▼
                         [Semantic & Concept RAG Scoring]
                         - Sentence-Transformers Local Embedding (all-MiniLM-L6-v2)
                         - Cosine Similarity scoring against JD
                         - Concept Similarity scoring against (Ranking, Evaluation, Production)
                                       │
                                       ▼
                             [Weighted Score Fusion]
                             - Semantic (20%), Structured (55%), Behavioral (15%), Quality (10%)
                             - Experience Curve & Trap Probability Penalties
                             - Scale/Behavioral boosts & Multi-domain Technical boosts
                             - Location & Availability multipliers (e.g. Pune/Noida/Delhi preferred, Willing to relocate)
                                       │
                                       ▼
                         [Post-Scoring Detailed Honeypot] (Top 300 candidates evaluated for traps/timeline sanity)
                                       │
                                       ▼
                         [Ranker, Tie-breaker & Validator]
                         - Ascending candidate_id as strict tie-breaker
                         - Automated reasoning generator
                                       │
                                       ▼
                           [Top 100 Selected Candidates]
```

### Detailed Execution Steps:
1. **Load & Validate Candidates**: Candidate records are read lazily/streamed from JSONL format to keep RAM footprint low (< 16 GB), validating each record's schema.
2. **Fast Pre-filter**: Cheap checks are performed to immediately filter out candidates claiming non-tech roles or failing the hard minimum years of experience threshold (e.g. 3 years).
3. **Parse JD Dynamically**: Extracts must-have skills, preferred experience range, and domain keywords from the job description.
4. **Structured Feature Scoring**: Calculates candidate-specific features (skills fuzzy matching, experience levels, education, certifications, and behavioral signals).
5. **Streaming Min-Heap Selection**: Instead of loading and scoring all 100k candidates semantically, a streaming min-heap keeps only the **Top 5,000 candidates** by structured score for the subsequent heavy embedding/reranking phase.
6. **RAG-Based Semantic Scoring**: Employs a local, offline SentenceTransformer model (`all-MiniLM-L6-v2`) to encode profile text summaries and job description.
7. **Concept Embedding Similarity**: Generates concept embeddings for *ranking*, *evaluation*, and *production* to evaluate candidates against specific advanced skills.
8. **Weighted Score Fusion**: Integrates all dimensions, applying penalties for experience mismatches, trap likelihoods, and availability/location issues.
9. **Detailed Honeypot & Quality Filtering (Post-Scoring Stage)**: Candidate profiles in the top 300 of initial scores undergo rigorous disqualification checks (timeline sanity, age vs career contradiction, skill explosion check, etc.). Failures are dropped or heavily penalized.
10. **Ranking & Tie-breaking**: Sorts candidates descending by final score. If scores are equal, candidate_id lexicographical order (ascending) is used as a strict tie-breaker.
11. **Submission Output & Validation**: Generates the Top 100 candidate submission CSV and runs validation scripts to guarantee perfect adherence to the rules.

---

## System Architecture & Hybrid Scoring

To ensure both robustness and deep understanding, the system uses a **multi-signal hybrid scoring pipeline**:

### 1. Ingestion & Quality Filtering (Honeypot Detection)
To prevent adversarial/synthetic profiles from skewing results, the ingestion layer evaluates profile sanity:
* **Experience Timeline Sanity Check**: Flagging candidates who claim years of experience exceeding the duration since their graduation.
* **Age vs. Career Contradiction**: Catching profiles where total working experience doesn't align with age.
* **Skill Explosion Check**: Identifying profiles containing unrealistic numbers of skills (e.g. 80+ skills).
* **Deduplication & Career Progression Checks**: Flags junior-to-executive leaps inside short timelines and removes identical profiles.
* Candidates receive a `quality_score` (0-1), and any candidates below a strict threshold are filtered out.

### 2. Structured Feature Scoring
* **Skills Matcher**: Tokenizes job description skills and candidate skills, applying fuzzy matching (`rapidfuzz`) to catch spelling variants (e.g. "ReactJS" vs "React.js").
* **Experience Scorer**: Evaluates required vs. actual years of experience and matches candidate job titles against the target role.
* **Education & Certifications**: Weighs degree level, field relevance, and verified industry certifications.
* **Behavioral Signals**: Maps platform telemetry (completeness, recruiter response rate, interview attendance, offer acceptance) to numeric weights.

### 3. Semantic RAG Scoring
* Employs a local HuggingFace `all-MiniLM-L6-v2` transformer (~80MB) to convert candidate profiles and the job description into vector embeddings.
* **Offline Execution**: Fully local inference—zero external API dependencies (no OpenAI, no Claude).
* **Batch Optimization**: Candidate profile summaries, skills, and titles are processed in batches of 512 to meet CPU execution constraints.

### 4. Rebalanced Score Fusion Formula
The suitability score uses rebalanced weights prioritizing technical dominance, combined with specific multipliers and penalty factors:

$$\text{Base Score} = 0.20 \times \text{Semantic Score} + 0.55 \times \text{Structured Score} + 0.15 \times \text{Behavioral Score} + 0.10 \times \text{Quality Score}$$

$$\text{Final Score} = \text{Base Score} \times \text{Experience Fit Modifier} \times \text{Trap Penalty} \times \text{Domain Boost} \times \text{Scale Boost} \times \text{Behavioral Boost} \times \text{Availability Mult} \times \text{Location Mult}$$

* **Experience Fit Modifier**: Penalty applied using a smooth dynamic curve if candidate experience is outside the ideal range (5–9 years).
* **Trap Penalty**: Deducts score proportional to the candidate's trap probability (identifying synthetic details/hype).
* **Domain Boost**: Up to 5% boost for true multi-domain technical experts (high scores across retrieval, ranking, evaluation, and production).
* **Availability & Location Multipliers**: Deducts score or disqualifies candidates based on location mismatch, unwillingness to relocate, platform inactivity (>90 days), or low response rates.

---

## Performance Constraints & Optimizations

* **Execution Time**: Under **5 minutes** for 100k candidate profiles on standard CPU-only hardware.
  - **Cold Start** (no cached embeddings, includes first-time local model download and embedding generation for the top 5,000 pre-selected candidates): **~189.8 seconds** (~3 minutes 10 seconds).
  - **Warm Start** (retrieving pre-computed/cached embeddings from `models/`): **~38.8 seconds**.
* **Memory Limits**: Operates within **16 GB RAM** via streaming files and pre-filtering candidate subsets before executing vector search.
* **Determinism**: Random seeds for `numpy` and `random` are fixed to guarantee 100% reproducible rankings.

---

## Setup & Execution (Reproducibility Guide)

This guide provides step-by-step instructions to replicate the pipeline, run tests, and validate the output.

### 1. Prerequisites & System Requirements
* **Python**: `Python 3.10` or higher (successfully developed and verified on `Python 3.12.3`).
* **Hardware**: Standard x86 or ARM CPU with at least 8 GB RAM (16 GB recommended for 100k candidate scaling).
* **Operating System**: Linux, macOS, or Windows.

### 2. Dependency List & Pinned Versions
The following external libraries are required and pinned inside `requirements.txt` for 100% environment reproducibility:
* `numpy==2.4.6` — Multi-signal vector metrics and array operations.
* `pandas==3.0.3` — Data extraction, manipulation, and CSV generation.
* `scikit-learn==1.9.0` — Similarity and distance metric helpers.
* `sentence-transformers==5.5.1` — Local encoding of text tokens to dense vectors.
* `RapidFuzz==3.14.5` — High-performance token/skill fuzzy matching.
* `faiss-cpu==1.14.3` — Dense vector similarity index search on CPU.
* `orjson==3.11.9` — Ultra-fast, memory-efficient candidate JSONL serialization.
* `pytest==9.0.3` — Test suite execution.

### 3. Setup Virtual Environment & Install Dependencies
Navigate to the root of the project (`Redrob-Codebase`) and run the commands matching your operating system:

#### On Linux / macOS:
```bash
# 1. Create a virtual environment named 'venv'
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Upgrade pip to the latest version
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt
```

#### On Windows (PowerShell or CMD):
```powershell
# 1. Create a virtual environment named 'venv'
python -m venv venv

# 2. Activate the virtual environment
# In PowerShell:
.\venv\Scripts\Activate.ps1
# In Command Prompt (CMD):
.\venv\Scripts\activate.bat

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt
```

### 4. Data Placement
Ensure that your input data files are located in the `data/` directory:
* **Candidate Pool**: `data/candidates.jsonl`
* **Job Description**: `data/job_description.md`

### 5. Running the Pipeline
With the virtual environment active, run the main entry point:
```bash
python main.py
```
This executes all phases of the pipeline:
1. Loads and parses the input datasets.
2. Runs the fast pre-filters and quality gating.
3. Extracts features on top candidates.
4. Downloads the SentenceTransformer model (`all-MiniLM-L6-v2`) locally to `models/` (first run only, fully cached/offline thereafter).
5. Calculates semantic & domain-concept scores.
6. Computes unified score fusion, sorts the results, and generates custom reasoning.
7. Saves the output to `outputs/submission.csv` (Top 100 candidates) and logs status information.

#### Additional CLI Options:
* Run with a subset of candidates to quickly test system functionality (e.g., 5,000 records):
  ```bash
  python main.py --limit 5000
  ```
* Specify a custom job description or output file path:
  ```bash
  python main.py --jd data/job_description.md --output outputs/submission.csv
  ```

### 6. Validating the Submission File
Verify that the output `submission.csv` strictly adheres to all challenge formatting rules (exactly 100 rows, required headers, correct columns, non-increasing score order, and proper lexicographical tie-breaks) by executing:
```bash
python "../Descriptions and protocol/validate_submission.py" outputs/submission.csv
```

### 7. Running Tests
Verify the complete functionality, safety gates, and scoring logic by running the automated test suite:
```bash
pytest
```
Or for a detailed verbose test run:
```bash
pytest -v
```
All unit and integration tests inside the `tests/` directory should pass successfully.
