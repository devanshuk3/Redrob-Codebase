# PRODUCT REQUIREMENTS DOCUMENT (PRD)

## Project

Intelligent Candidate Discovery & Ranking System

## Competition

Redrob Intelligent Candidate Discovery & Ranking Challenge

---

# 1. PROJECT OBJECTIVE

Develop a candidate ranking system capable of identifying and ranking the 100 best candidates from a pool of 100,000 candidate profiles for a provided job description.

The system must produce a reproducible ranking that satisfies all competition requirements and can withstand automated validation, code reproduction, manual review, and technical interview evaluation.

---

# 2. PRIMARY DELIVERABLES

The project must produce:

### 2.1 Submission CSV

A CSV file containing:

* Top 100 candidates only
* Unique ranks from 1–100
* Candidate score
* Candidate reasoning

Required columns:

* candidate_id
* rank
* score
* reasoning

### 2.2 Source Code Repository

Repository must contain:

* Complete source code
* README
* Dependency definitions
* Reproducibility instructions
* Metadata file
* Any required artifacts

### 2.3 Sandbox Environment

Hosted environment capable of:

* Running the ranking system
* Accepting candidate data
* Producing ranked output
* Demonstrating reproducibility

### 2.4 Submission Metadata

Required metadata:

* Team name
* Team members
* Contact details
* GitHub repository
* Sandbox link
* AI tool declaration
* Compute environment summary

---

# 3. FUNCTIONAL REQUIREMENTS

The ranking system shall:

### FR-1 Candidate Evaluation

Evaluate candidate suitability against the provided job description.

### FR-2 Candidate Ranking

Produce an ordered ranking of candidates.

### FR-3 Candidate Scoring

Generate a numerical score representing candidate suitability.

### FR-4 Top-100 Selection

Return only the best 100 candidates.

### FR-5 Ranking Consistency

Maintain rank and score consistency throughout the output.

### FR-6 Candidate Reasoning

Generate concise reasoning for candidate placement.

### FR-7 Candidate Validation

Process only valid candidate records.

### FR-8 Behavioral Signal Utilization

Incorporate candidate behavioral indicators where applicable.

### FR-9 Reproducibility

Generate identical results when executed under the same conditions.

### FR-10 Offline Operation

Operate without external services during ranking execution.

---

# 4. NON-FUNCTIONAL REQUIREMENTS

## Performance

The ranking process must:

* Complete within 5 minutes
* Run on CPU-only hardware
* Operate within 16 GB RAM
* Use no more than 5 GB intermediate storage

## Reliability

The system must:

* Produce deterministic results
* Avoid crashes during processing
* Handle malformed records gracefully

## Scalability

The system must process:

* 100,000 candidate records
* Within competition constraints

## Reproducibility

Outputs must be reproducible from source code alone.

---

# 5. DATA SOURCES

The system shall use:

### Candidate Dataset

* candidates.jsonl.gz
* candidates.jsonl

### Candidate Schema

* candidate_schema.json

### Behavioral Signal Definitions

* redrob_signals_doc

### Job Description

* job_description.md

---

# 6. CANDIDATE ASSESSMENT DIMENSIONS

The system should consider:

## Candidate Profile Information

* Skills
* Experience
* Education
* Employment history
* Projects
* Certifications
* Role relevance

## Behavioral Indicators

* Profile completeness
* Platform activity
* Recruiter engagement
* Response behavior
* Interview attendance
* Offer acceptance behavior
* Work availability
* Notice period
* Verification status
* Professional network activity

---

# 7. DATA QUALITY REQUIREMENTS

The system must account for:

### Incomplete Profiles

Candidates with missing information.

### Contradictory Profiles

Profiles containing inconsistencies.

### Suspicious Profiles

Profiles exhibiting unrealistic characteristics.

### Synthetic Profiles

Artificially generated candidate data.

### Honeypot Profiles

Candidates intentionally inserted to identify weak ranking systems.

---

# 8. HONEYPOT COMPLIANCE

The solution must minimize ranking of honeypot candidates.

Requirements:

* Detect suspicious profile characteristics
* Avoid impossible career histories
* Avoid impossible experience timelines
* Avoid unrealistic expertise claims

Failure Condition:

* Honeypot rate greater than 10% in Top 100

Result:

* Disqualification

---

# 9. OUTPUT REQUIREMENTS

## CSV Requirements

Must:

* Be UTF-8 encoded
* Contain exactly 100 rows
* Include required columns
* Use valid candidate IDs
* Use unique candidate IDs
* Use unique ranks

## Ranking Requirements

* Rank 1 = best candidate
* Rank 100 = lowest selected candidate
* Scores must be non-increasing
* Ties allowed
* Ranks must remain unique

---

# 10. REASONING REQUIREMENTS

Each reasoning entry should:

* Reference actual candidate facts
* Reference job requirements
* Reflect ranking position
* Avoid unsupported claims
* Avoid hallucinations
* Remain concise

Reasoning should not:

* Contain fabricated information
* Contradict candidate data
* Contradict candidate rank

---

# 11. DEVELOPMENT COMPLIANCE REQUIREMENTS

## Allowed

* AI-assisted development
* Local models
* Traditional software engineering
* Feature engineering
* Statistical methods
* Information retrieval techniques
* Machine learning techniques

## Mandatory

* Honest AI tool disclosure
* Ability to explain all system components
* Ability to defend methodology
* Ability to reproduce results

---

# 12. PROHIBITED DURING RANKING EXECUTION

The ranking step must not:

* Use hosted LLM APIs
* Use external AI APIs
* Use external web services
* Use network calls
* Use GPU resources

Examples:

* OpenAI API
* Gemini API
* Claude API
* Cohere API

---

# 13. REPOSITORY REQUIREMENTS

Repository must include:

## Documentation

* README.md
* Setup instructions
* Execution instructions

## Source Code

* Complete implementation
* No hidden steps
* No manual ranking modifications

## Dependency Information

* requirements.txt
  or
* pyproject.toml

## Metadata

* submission_metadata.yaml

## Reproduction Instructions

A single documented command capable of generating the final submission.

---

# 14. SANDBOX REQUIREMENTS

The sandbox must:

* Be publicly accessible or shareable
* Execute the ranking workflow
* Produce ranked output
* Demonstrate reproducibility
* Operate within competition limits

Acceptable platforms include:

* HuggingFace Spaces
* Streamlit Cloud
* Replit
* Google Colab
* Binder
* Docker deployment

---

# 15. EVALUATION STAGES

## Stage 1

Format Validation

Focus:

* File correctness
* Submission compliance

## Stage 2

Automated Scoring

Focus:

* Ranking quality

## Stage 3

Code Reproduction

Focus:

* Reproducibility
* Constraint compliance
* Honeypot rate

## Stage 4

Manual Review

Focus:

* Reasoning quality
* Repository quality
* Methodology coherence
* Git history authenticity

## Stage 5

Technical Defense Interview

Focus:

* Architecture understanding
* Engineering decisions
* Design justification

---

# 16. SUCCESS CRITERIA

A successful submission must:

* Pass validation
* Produce valid rankings
* Avoid honeypot disqualification
* Satisfy compute limits
* Be reproducible
* Include complete documentation
* Include valid sandbox access
* Survive manual review
* Survive technical interview

---

# 17. TEAM CHECKLIST

Before submission verify:

□ Exactly 100 candidates selected

□ CSV format validated

□ Scores correctly ordered

□ Ranks unique

□ Candidate IDs valid

□ Reasoning generated

□ Repository complete

□ Metadata completed

□ Sandbox operational

□ AI tools declared

□ Compute constraints satisfied

□ Reproduction instructions verified

□ Honeypot risk reviewed

□ Team prepared for interview

□ Final submission generated

□ Submission validated successfully

---

# FINAL PROJECT GOAL

Deliver a reproducible, explainable, constraint-compliant candidate ranking system that accurately identifies the highest-quality candidates for the target job description while avoiding synthetic traps, maintaining ranking integrity, and satisfying all competition evaluation stages.
