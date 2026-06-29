"""
sandbox/app.py
===============
Streamlit sandbox UI for the RedRob candidate ranking pipeline.

This satisfies spec Section 10.5 (hosted sandbox / demo link). It contains
NO ranking logic of its own — it's a thin wrapper that:

  1. Lets you upload a candidates.jsonl sample
  2. Lets you optionally override the job description (defaults to the
     bundled data/job_description.md)
  3. Runs the real pipeline exactly as Stage 3 will reproduce it:
         python main.py --jd <jd_path> --output <output_path>
  4. Shows the resulting submission.csv, elapsed time, and a download button

Run locally (from the repo root):
    streamlit run sandbox/app.py

Deploy on Streamlit Cloud:
    Point the app at this file. Make sure `streamlit` is in requirements.txt
    (it isn't there by default — see README note below).
"""

import os
import subprocess
import sys
import tempfile
import time

import pandas as pd
import streamlit as st

# This file lives in sandbox/, so the repo root is one level up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")
CANDIDATES_PATH = os.path.join(REPO_ROOT, "data", "candidates.jsonl")
DEFAULT_JD_PATH = os.path.join(REPO_ROOT, "data", "job_description.md")
COMPUTE_BUDGET_SECONDS = 300  # 5-minute limit from the submission spec

st.set_page_config(page_title="RedRob Ranking Sandbox", layout="wide")
st.title("RedRob Candidate Ranking — Sandbox")
st.caption(
    "Runs the real pipeline (`main.py`) end-to-end against a small candidate "
    "sample. This wrapper doesn't reimplement any ranking logic — it just "
    "invokes your existing code, the same way Stage 3 will reproduce it."
)

with st.expander("Things to know about this pipeline before you run it", expanded=False):
    st.markdown(
        """
- **`build_submission()` hard-requires exactly 100 output rows.** If fewer
  than 100 candidates survive pre-filtering and quality checks, the
  pipeline raises a `ValueError` instead of returning a partial ranking.
  Upload **at least 100** candidates (ideally a good amount more, since
  pre-filtering removes some) to see a successful run here.
- **First run needs internet access.** `models/all-MiniLM-L6-v2/` isn't
  checked into the repo, so on a fresh machine (including a fresh Streamlit
  Cloud deploy) the pipeline downloads that model from Hugging Face on
  first use. That's a one-time setup cost for *this sandbox* — it is not
  the network-off ranking step Stage 3 reproduces, so don't read a
  successful sandbox run as proof of that constraint.
"""
    )

st.subheader("1. Candidates")
candidates_file = st.file_uploader(
    "Upload a candidates.jsonl sample (≥100 records recommended)", type=["jsonl"]
)

st.subheader("2. Job description")
use_default_jd = st.checkbox("Use the bundled data/job_description.md", value=True)
jd_text = ""
if use_default_jd:
    if os.path.exists(DEFAULT_JD_PATH):
        with open(DEFAULT_JD_PATH, "r", encoding="utf-8") as f:
            st.text_area("Bundled JD (read-only preview)", f.read(), height=150, disabled=True)
    else:
        st.error(f"Bundled JD not found at {DEFAULT_JD_PATH}")
else:
    jd_text = st.text_area("Paste a job description", height=200)

run = st.button("Run pipeline (python main.py)", type="primary", width="stretch")

if run:
    if not candidates_file:
        st.error("Please upload a candidates.jsonl file.")
        st.stop()
    if not use_default_jd and not jd_text.strip():
        st.error("Please paste a job description, or use the bundled one.")
        st.stop()

    # 1. Write the upload to the exact path main.py reads from (Config.CANDIDATES_FILE).
    #    There's no --candidates CLI flag — the path is hardcoded in src/utils/config.py.
    os.makedirs(os.path.dirname(CANDIDATES_PATH), exist_ok=True)
    with open(CANDIDATES_PATH, "wb") as f:
        f.write(candidates_file.getvalue())

    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        n_lines = sum(1 for line in f if line.strip())
    if n_lines < 100:
        st.warning(
            f"Uploaded file has {n_lines} candidates. The pipeline requires "
            f"exactly 100 output rows after filtering — with fewer than 100 "
            f"input candidates this will very likely raise a ValueError below."
        )

    # 2. Resolve the JD path; write pasted text to a temp file if not using the default.
    if use_default_jd:
        jd_path_to_use = DEFAULT_JD_PATH
    else:
        tmp_jd = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        )
        tmp_jd.write(jd_text)
        tmp_jd.close()
        jd_path_to_use = tmp_jd.name

    output_path = os.path.join(tempfile.mkdtemp(), "submission.csv")

    # 3. Run the real pipeline — the exact command from the repo's README / Stage 3 reproduction.
    cmd = [sys.executable, MAIN_PY, "--jd", jd_path_to_use, "--output", output_path]
    st.info(f"Running: `{' '.join(cmd)}`")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=COMPUTE_BUDGET_SECONDS,
        )
        timed_out = False
    except subprocess.TimeoutExpired as e:
        result = e
        timed_out = True
    elapsed = time.time() - start

    log_text = (result.stdout or "") + (result.stderr or "")
    with st.expander("Pipeline logs", expanded=not timed_out and getattr(result, "returncode", 1) != 0):
        st.code(log_text or "(no output captured)", language="bash")

    if timed_out:
        st.error(f"Exceeded the {COMPUTE_BUDGET_SECONDS}s compute budget — pipeline timed out.")
        st.stop()

    if result.returncode != 0:
        st.error(f"Pipeline exited with code {result.returncode} after {elapsed:.1f}s. See logs above.")
        st.stop()

    st.success(f"Pipeline finished in {elapsed:.1f}s.")
    if elapsed > COMPUTE_BUDGET_SECONDS:
        st.warning("This exceeded the 5-minute compute budget required by the spec.")

    if not os.path.exists(output_path):
        st.error(f"Pipeline reported success but no output file was found at {output_path}.")
        st.stop()

    df = pd.read_csv(output_path)
    st.subheader("Ranked output")
    st.dataframe(df, width="stretch")

    st.download_button(
        "Download submission.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="submission.csv",
        mime="text/csv",
        type="primary",
    )
else:
    st.info("Upload a candidates file, then click **Run pipeline**.")