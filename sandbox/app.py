"""
sandbox/app.py
===============
Streamlit sandbox UI for the RedRob candidate ranking pipeline.

This satisfies spec Section 10.5 (hosted sandbox / demo link). It contains
NO ranking logic of its own — it's a thin wrapper that:

  1. Auto-detects data/candidates.jsonl if present; otherwise lets you
     upload a candidates.jsonl sample
  2. Lets you optionally override the job description (defaults to the
     bundled data/job_description.md)
  3. Runs the real pipeline exactly as Stage 3 will reproduce it:
         python main.py --jd <jd_path> --output <output_path>
     All outputs (submission.csv, candidate_scores.csv, debug CSVs) are
     written to the standard outputs/ directory — identical to running
     main.py from the terminal.
  4. Streams pipeline logs in real-time (matching the terminal output)
     with a live elapsed-time timer
  5. Shows the resulting submission.csv and a download button

Run locally (from the repo root):
    streamlit run sandbox/app.py

Deploy on Streamlit Cloud:
    Point the app at this file. Make sure `streamlit` is in requirements.txt
    (it isn't there by default — see README note below).
"""

import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

import pandas as pd
import streamlit as st

# This file lives in sandbox/, so the repo root is one level up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")
CANDIDATES_PATH = os.path.join(REPO_ROOT, "data", "candidates.jsonl")
DEFAULT_JD_PATH = os.path.join(REPO_ROOT, "data", "job_description.md")
DEFAULT_OUTPUT_PATH = os.path.join(REPO_ROOT, "outputs", "submission.csv")
COMPUTE_BUDGET_SECONDS = 300  # 5-minute limit from the submission spec

st.set_page_config(page_title="RedRob Ranking Sandbox", layout="wide")
st.title("RedRob Candidate Ranking — Sandbox")
st.caption(
    "Runs the real pipeline (`main.py`) end-to-end against candidate data. "
    "This wrapper doesn't reimplement any ranking logic — it just "
    "invokes your existing code, the same way Stage 3 will reproduce it."
)

with st.expander("Things to know about this pipeline before you run it", expanded=False):
    st.markdown(
        """
- **`build_submission()` hard-requires exactly 100 output rows.** If fewer
  than 100 candidates survive pre-filtering and quality checks, the
  pipeline raises a `ValueError` instead of returning a partial ranking.
  Ensure **at least 100** candidates (ideally a good amount more, since
  pre-filtering removes some) to see a successful run here.
- **First run needs internet access.** `models/all-MiniLM-L6-v2/` isn't
  checked into the repo, so on a fresh machine (including a fresh Streamlit
  Cloud deploy) the pipeline downloads that model from Hugging Face on
  first use. That's a one-time setup cost for *this sandbox* — it is not
  the network-off ranking step Stage 3 reproduces, so don't read a
  successful sandbox run as proof of that constraint.
- **All outputs are written to `outputs/`.** This includes `submission.csv`,
  `candidate_scores.csv`, debug CSVs in `outputs/debug/`, and logs in
  `outputs/logs/` — exactly the same as running `python main.py` directly.
"""
    )

# ── 1. Candidates ──────────────────────────────────────────────────────────
st.subheader("1. Candidates")

# Check if data/candidates.jsonl already exists on disk
candidates_file_exists = os.path.exists(CANDIDATES_PATH) and os.path.getsize(CANDIDATES_PATH) > 0

if candidates_file_exists:
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        existing_line_count = sum(1 for line in f if line.strip())
    st.success(
        f"✅ Auto-detected `data/candidates.jsonl` ({existing_line_count:,} candidates). "
        f"The pipeline will use this file automatically."
    )
    use_existing = st.checkbox("Use auto-detected candidates file", value=True)
    if use_existing:
        candidates_file = None  # Signal: no upload needed, file already in place
    else:
        candidates_file = st.file_uploader(
            "Upload a different candidates.jsonl file (≥100 records recommended)", type=["jsonl"]
        )
else:
    st.info("No `data/candidates.jsonl` found — please upload one below.")
    use_existing = False
    candidates_file = st.file_uploader(
        "Upload a candidates.jsonl sample (≥100 records recommended)", type=["jsonl"]
    )

# ── 2. Job description ────────────────────────────────────────────────────
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

# ── 3. Run pipeline ───────────────────────────────────────────────────────
run = st.button("Run pipeline (python main.py)", type="primary", width="stretch")

if run:
    # Validate candidate source
    if use_existing and candidates_file_exists:
        # File already in place — nothing to write
        pass
    elif candidates_file is not None:
        # Write the uploaded file to the path main.py expects
        os.makedirs(os.path.dirname(CANDIDATES_PATH), exist_ok=True)
        with open(CANDIDATES_PATH, "wb") as f:
            f.write(candidates_file.getvalue())
    else:
        st.error("Please upload a candidates.jsonl file or ensure one exists in `data/`.")
        st.stop()

    # Count candidates and warn if too few
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        n_lines = sum(1 for line in f if line.strip())
    if n_lines < 100:
        st.warning(
            f"Candidates file has {n_lines} records. The pipeline requires "
            f"exactly 100 output rows after filtering — with fewer than 100 "
            f"input candidates this will very likely raise a ValueError below."
        )

    # Validate JD
    if not use_default_jd and not jd_text.strip():
        st.error("Please paste a job description, or use the bundled one.")
        st.stop()

    # Resolve JD path
    if use_default_jd:
        jd_path_to_use = DEFAULT_JD_PATH
    else:
        tmp_jd = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        )
        tmp_jd.write(jd_text)
        tmp_jd.close()
        jd_path_to_use = tmp_jd.name

    # Use the standard output path so all outputs (submission, debug CSVs,
    # candidate_scores.csv) land in the same outputs/ directory as a
    # terminal run of main.py.
    output_path = DEFAULT_OUTPUT_PATH

    # Build the command — identical to running main.py from the terminal
    cmd = [sys.executable, MAIN_PY, "--jd", jd_path_to_use, "--output", output_path]
    st.info(f"Running: `{' '.join(cmd)}`")

    # ── Live timer + log streaming ────────────────────────────────────
    st.subheader("Pipeline Execution")

    # Reserve a container for results ABOVE the logs — will be populated
    # after the pipeline finishes so results appear before logs.
    results_container = st.container()

    # Timer display
    timer_placeholder = st.empty()

    # Log display
    st.markdown("**Pipeline Logs**")
    log_container = st.empty()
    log_lines = []

    start = time.time()
    timed_out = False
    return_code = None

    def _format_elapsed(seconds):
        """Format elapsed seconds as MM:SS."""
        mins, secs = divmod(int(seconds), 60)
        return f"{mins:02d}:{secs:02d}"

    def _reader_thread(pipe, q):
        """Background thread that reads stdout line-by-line into a queue."""
        try:
            for line in iter(pipe.readline, ""):
                q.put(line.rstrip("\n"))
        except ValueError:
            pass  # pipe closed
        finally:
            q.put(None)  # sentinel: signals EOF

    try:
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for unified log stream
            text=True,
            bufsize=1,  # Line-buffered
            env={**os.environ, "PYTHONUNBUFFERED": "1"},  # Force unbuffered Python output
        )

        # Start a background thread to read stdout without blocking
        line_queue = queue.Queue()
        reader = threading.Thread(
            target=_reader_thread, args=(process.stdout, line_queue), daemon=True
        )
        reader.start()

        # Main loop: tick every second, drain new log lines, update timer
        while True:
            elapsed = time.time() - start
            timer_placeholder.metric("⏱️ Elapsed Time", _format_elapsed(elapsed))

            # Check timeout
            if elapsed > COMPUTE_BUDGET_SECONDS:
                process.kill()
                timed_out = True
                break

            # Drain all available lines from the queue (non-blocking)
            new_lines = False
            while True:
                try:
                    line = line_queue.get_nowait()
                except queue.Empty:
                    break
                if line is None:
                    # EOF sentinel — process finished writing
                    break
                log_lines.append(line)
                new_lines = True

            if new_lines:
                log_container.code("\n".join(log_lines), language="log")

            # Check if process has exited AND queue is drained
            if process.poll() is not None:
                # Drain any final lines still in the queue
                while True:
                    try:
                        line = line_queue.get_nowait()
                    except queue.Empty:
                        break
                    if line is None:
                        break
                    log_lines.append(line)
                if log_lines:
                    log_container.code("\n".join(log_lines), language="log")
                break

            # Sleep ~1 second so the timer updates smoothly
            time.sleep(1)

        return_code = process.returncode

    except Exception as e:
        st.error(f"Pipeline execution failed: {e}")
        st.stop()

    elapsed = time.time() - start

    # Final timer update
    timer_placeholder.metric("⏱️ Total Time", _format_elapsed(elapsed))

    # Final log display
    if log_lines:
        log_container.code("\n".join(log_lines), language="log")
    else:
        log_container.code("(no output captured)", language="log")

    # ── Handle results (rendered ABOVE logs via results_container) ─────
    if timed_out:
        results_container.error(f"Exceeded the {COMPUTE_BUDGET_SECONDS}s compute budget — pipeline timed out.")
        st.stop()

    if return_code != 0:
        results_container.error(f"Pipeline exited with code {return_code} after {elapsed:.1f}s. See logs below.")
        st.stop()

    with results_container:
        st.success(f"✅ Pipeline finished successfully in {_format_elapsed(elapsed)} ({elapsed:.1f}s).")
        if elapsed > COMPUTE_BUDGET_SECONDS:
            st.warning("This exceeded the 5-minute compute budget required by the spec.")

        if not os.path.exists(output_path):
            st.error(f"Pipeline reported success but no output file was found at {output_path}.")
            st.stop()

        df = pd.read_csv(output_path)
        st.subheader("Ranked Output")
        st.dataframe(df, width="stretch")

        st.download_button(
            "Download submission.csv",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="submission.csv",
            mime="text/csv",
            type="primary",
        )
else:
    if candidates_file_exists:
        st.info("Auto-detected candidates file is ready. Click **Run pipeline** to start.")
    else:
        st.info("Upload a candidates file, then click **Run pipeline**.")