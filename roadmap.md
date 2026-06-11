## Project Roadmap: Intelligent Candidate Discovery & Ranking System


### Understanding the Core Problem

Before jumping to implementation, internalize what this actually is: given a job description and 100k candidate profiles, rank the top 100. The hard constraints are:

- **No external APIs during ranking** (no OpenAI, no Claude, nothing) 

- CPU-only, 16GB RAM, 5 min runtime, 5GB storage 

- Must detect and reject honeypot/synthetic profiles 

- Results must be deterministic and reproducible 

This means your pipeline has to be **fully local** — local models, local embeddings, local inference if any.


### Recommended Architecture (Hybrid Scoring Pipeline)

Don't rely on a single approach. The winning strategy is a **multi-signal scoring pipeline** that combines:

1. **Semantic similarity** (RAG/embedding-based) — how well does the candidate match the JD semantically 

2. **Structured feature scoring** — skills overlap, years of experience, education level, certifications 

3. **Behavioral signal scoring** — platform activity, recruiter engagement, offer acceptance, etc. 

4. **Data quality / honeypot filtering** — eliminate suspicious profiles before ranking 

Final score = weighted combination of all signals.


### Phase-wise Roadmap


#### Phase 0 — Setup & Data Exploration (Day 1–2)

**All members involved**

- Set up the repo with the structure defined below 

- Everyone reads the PRD, job description, `candidate\_schema.json`, and `redrob\_signals\_doc` thoroughly 

- Do exploratory data analysis (EDA) on a sample of candidates — understand field distributions, null rates, data quality 

- Define the scoring weights together as a team decision (you can tune later) 

- Agree on the final output schema and column formats upfront 




#### Phase 1 — Data Ingestion & Quality Filtering (Day 2–4)

**Owner: Member A**

This is the foundation. If garbage goes in, garbage comes out.

Tasks:

- Write the data loader for `candidates.jsonl.gz` — streaming read to stay within RAM 

- Schema validation — reject records missing required fields 

- **Honeypot detection** — this is critical. Rules to implement: 

  - Experience timeline sanity check (e.g., graduated in 2020 but claims 15 years of experience) 

  - Age vs. career length contradiction 

  - Skill count explosion (someone claiming 80+ skills is suspicious) 

  - Identical or near-identical profiles (deduplication) 

  - Unrealistic career progression (junior → CTO in 1 year) 

  - Employment gap contradictions 

- Flag suspicious candidates with a `quality\_score` (0–1), filter out anything below a threshold 

- Output: cleaned candidate list with quality flags 


#### Phase 2 — Structured Feature Engineering (Day 3–5)

**Owner: Member B**

This is your deterministic, fast, interpretable layer — works without any ML.

Tasks:

- **Skills matching**: tokenize JD skills, compute overlap ratio with candidate skills. Use fuzzy matching (rapidfuzz) for variations like "ReactJS" vs "React.js" 

- **Experience scoring**: required years vs. candidate years, role relevance (titles matching JD) 

- **Education scoring**: degree level weighting, field relevance 

- **Recency scoring**: recent experience matters more than decade-old roles 

- **Certification scoring**: relevant certs get a bonus 

- **Behavioral signal scoring**: map each signal from `redrob\_signals\_doc` to a numeric value — profile completeness, interview attendance, offer acceptance rate, recruiter responsiveness 

- Output: a structured feature vector per candidate with sub-scores 






#### Phase 3 — RAG-Based Semantic Scoring (Day 3–6)

**Owner: Member C**

This is the intelligent layer that catches what structured features miss.

How RAG fits here — the idea is to treat the job description as the "query" and candidate profiles as "documents," then retrieve the most semantically relevant candidates using embeddings.

Implementation:

- Use a **local sentence-transformer model** (e.g., `all-MiniLM-L6-v2` from HuggingFace) — it's small (~80MB), fast on CPU, and good enough 

- Encode the job description into an embedding vector 

- For each candidate, concatenate their key text fields (skills, job titles, summary, experience descriptions) into a single text blob and encode it 

- Compute **cosine similarity** between JD embedding and candidate embedding 

- This gives you a semantic relevance score (0–1) 

Key concern: 100k candidates × embedding computation might be slow. Optimizations:

- Use batch encoding (sentence-transformers supports this natively) 

- Run in batches of 512 or 1024 

- Pre-filter candidates using structured scores first (e.g., take top 20k from Phase 2), then do embeddings only on that subset — this keeps you within 5 minutes 

- Use `faiss` (CPU version) for fast nearest-neighbor search if needed 

Output: semantic similarity score per candidate


#### Phase 4 — Score Fusion & Final Ranking (Day 6–7)

**Owner: Member D**

Combine all signals into a final score and produce the submission CSV.

Tasks:

- Define weighted formula. Example starting point: 

- ```
`final\_score = 0.35 × semantic\_score            + 0.30 × structured\_feature\_score            + 0.20 × behavioral\_score            + 0.15 × quality\_score`
```

- These weights are tunable — the team should decide together based on what the JD emphasizes. 

- Sort by final score descending, take top 100 

- Assign ranks 1–100 

- Generate reasoning strings: template-based is fine, e.g., *"Candidate has 6 years of relevant experience in X, strong skills overlap (Y, Z), and high platform engagement score."* — pull actual candidate facts, don't hallucinate 

- Validate output: exactly 100 rows, unique ranks, non-increasing scores, valid IDs 

- UTF-8 encode and export CSV 


#### Phase 5 — Integration, Testing & Reproducibility (Day 7–9)

**All members**

- Wire all phases into a single `run.py` or `main.py` that goes from raw data to final CSV with one command 

- Set all random seeds (`random.seed(42)`, `numpy.seed(42)`) for determinism 

- Test full pipeline end to end — time it, memory profile it 

- Confirm it runs under 5 minutes, under 16GB RAM 

- Test edge cases: malformed records, missing fields 

- Member A verifies honeypot filtering is working 

- Member D validates the final CSV format strictly 


#### Phase 6 — Sandbox & Documentation (Day 9–10)

**Owner: Member D + Member A**

- Deploy to HuggingFace Spaces or Streamlit Cloud 

- Write `README.md` — setup, execution, methodology explanation 

- Fill `submission\_metadata.yaml` 

- Write a `METHODOLOGY.md` explaining every design decision (you'll need this for the technical defense interview) 

- Verify the single-command reproduction works on a clean environment 


### Repository Structure

```
`candidate-ranking/`

`│`

`├── data/                          \# gitignored — local data only`

`│   ├── candidates.jsonl.gz`

`│   ├── candidate\_schema.json`

`│   ├── redrob\_signals\_doc`

`│   └── job\_description.md`

`│`

`├── src/`

`│   ├── ingestion/                 \# Member A`

`│   │   ├── \_\_init\_\_.py`

`│   │   ├── loader.py              \# streaming data loader`

`│   │   ├── validator.py           \# schema validation`

`│   │   └── honeypot\_filter.py     \# suspicious profile detection`

`│   │`

`│   ├── features/                  \# Member B`

`│   │   ├── \_\_init\_\_.py`

`│   │   ├── skills\_scorer.py`

`│   │   ├── experience\_scorer.py`

`│   │   ├── education\_scorer.py`

`│   │   ├── behavioral\_scorer.py`

`│   │   └── feature\_pipeline.py    \# orchestrates all sub-scorers`

`│   │`

`│   ├── semantic/                  \# Member C`

`│   │   ├── \_\_init\_\_.py`

`│   │   ├── embedder.py            \# sentence-transformer wrapper`

`│   │   ├── similarity.py          \# cosine sim / faiss search`

`│   │   └── semantic\_pipeline.py`

`│   │`

`│   ├── ranking/                   \# Member D`

`│   │   ├── \_\_init\_\_.py`

`│   │   ├── score\_fusion.py        \# weighted combination`

`│   │   ├── ranker.py              \# sorting, top-100 selection`

`│   │   ├── reasoning\_generator.py \# generates reasoning strings`

`│   │   └── output\_validator.py    \# validates final CSV`

`│   │`

`│   └── utils/`

`│       ├── \_\_init\_\_.py`

`│       ├── config.py              \# weights, thresholds, seeds`

`│       └── logging\_utils.py`

`│`

`├── models/                        \# local model files (gitignored if large)`

`│   └── all-MiniLM-L6-v2/`

`│`

`├── outputs/`

`│   └── submission.csv             \# final output`

`│`

`├── notebooks/                     \# EDA, experimentation — not part of pipeline`

`│   ├── eda.ipynb`

`│   └── weight\_tuning.ipynb`

`│`

`├── tests/`

`│   ├── test\_ingestion.py`

`│   ├── test\_features.py`

`│   ├── test\_semantic.py`

`│   └── test\_ranking.py`

`│`

`├── sandbox/                       \# Streamlit/HuggingFace app`

`│   └── app.py`

`│`

`├── main.py                        \# single entrypoint — runs full pipeline`

`├── requirements.txt`

`├── submission\_metadata.yaml`

`├── README.md`

`└── METHODOLOGY.md`
```


### Member Responsibility Summary

| **Member** | **Owns** | **Key deliverable** |
| :-: | :-: | :-: |
| A | Ingestion + Honeypot filtering | Clean, validated candidate list |
| B | Structured feature engineering | Feature score vector per candidate |
| C | Semantic/RAG scoring | Embedding similarity scores |
| D | Score fusion + output + sandbox | Final CSV + deployment |

Everyone contributes to Phase 0 (EDA) and Phase 5 (integration/testing). All four need to understand the full pipeline well enough to defend it in the technical interview — so don't let members work in complete silos.



### Tech Stack Recommendation

| **Purpose** | **Library** |
| :-: | :-: |
| Data loading | `ijson` (streaming JSON) |
| Fuzzy matching | `rapidfuzz` |
| Embeddings | `sentence-transformers` |
| Vector search | `faiss-cpu` |
| Numerical ops | `numpy`, `pandas` |
| Sandbox UI | `streamlit` |
| Testing | `pytest` |


### Critical Things to Not Screw Up

- **No API calls during ranking** — the embedder must use a locally downloaded model, not pulled at runtime 

- **Honeypots** — if \>10% of your top 100 are honeypots, you're disqualified. Take this seriously 

- **Determinism** — fix all seeds, avoid any randomness that isn't seeded 

- **Reasoning** — pull actual facts from the candidate data, not generic fluff. The manual review stage checks this 

- **Single command run** — `python main.py` should do everything from raw data to `outputs/submission.csv` 

