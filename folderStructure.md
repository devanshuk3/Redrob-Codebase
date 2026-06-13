candidate-ranking/
│
├── data/                          # gitignored — local data only
│   ├── candidates.jsonl.gz
│   ├── candidate_schema.json
│   ├── redrob_signals_doc
│   └── job_description.md
│
├── src/
│   ├── ingestion/                 # Member A 
│   │   ├── __init__.py
│   │   ├── loader.py              # streaming data loader
│   │   ├── validator.py           # schema validation
│   │   └── honeypot_filter.py     # suspicious profile detection
│   │
│   ├── features/                  # Member B
│   │   ├── __init__.py
│   │   ├── skills_scorer.py
│   │   ├── experience_scorer.py
│   │   ├── education_scorer.py
│   │   ├── behavioral_scorer.py
│   │   └── feature_pipeline.py    # orchestrates all sub-scorers
│   │
│   ├── semantic/                  # Member C
│   │   ├── __init__.py
│   │   ├── embedder.py            # sentence-transformer wrapper
│   │   ├── similarity.py          # cosine sim / faiss search
│   │   └── semantic_pipeline.py
│   │
│   ├── ranking/                   # Member D
│   │   ├── __init__.py
│   │   ├── score_fusion.py        # weighted combination
│   │   ├── ranker.py              # sorting, top-100 selection
│   │   ├── reasoning_generator.py # generates reasoning strings
│   │   └── output_validator.py    # validates final CSV
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # weights, thresholds, seeds
│       └── logging_utils.py
│
├── models/                        # local model files (gitignored if large)
│   └── all-MiniLM-L6-v2/
│
├── outputs/
│   └── submission.csv             # final output
│
├── notebooks/                     # EDA, experimentation — not part of pipeline
│   ├── eda.ipynb
│   └── weight_tuning.ipynb
│   
├── tests/
│   ├── test_ingestion.py
│   ├── test_features.py
│   ├── test_semantic.py
│   └── test_ranking.py
│
├── sandbox/                       # Streamlit/HuggingFace app
│   └── app.py
│
├── main.py                        # single entrypoint — runs full pipeline
├── requirements.txt
├── submission_metadata.yaml
├── README.md
└── METHODOLOGY.md