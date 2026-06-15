# Use a stable, lightweight Python base image
FROM python:3.10-slim

# Set environment variables for clean logs and offline execution
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Install libgomp1 (required for faiss-cpu to run on Linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Pre-download and cache the sentence-transformer model during the build stage.
# This saves the model directly to the expected cache path (/app/models/all-MiniLM-L6-v2)
# so the container can run 100% offline without needing internet access.
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); model.save('/app/models/all-MiniLM-L6-v2')"

# Copy the application source code, tests, and entrypoint script
COPY src/ ./src/
COPY tests/ ./tests/
COPY main.py .

# Create placeholder directories for input data and outputs.
# These will be mapped to the host directory at runtime via volumes.
RUN mkdir -p data outputs

# Set environment variable to force HuggingFace library to run offline
ENV TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

# Default command to run the test suite first, and if they pass, run the main ranking pipeline.
CMD ["sh", "-c", "pytest && python main.py"]


