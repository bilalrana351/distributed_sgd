# Phase 4 – Docker image for hybrid distributed SGD training
#
# Builds a reproducible Linux environment that runs the same src/ code
# used in Phases 1–3 on the Windows host.
#
# Build:
#   docker build -t pdc-hybrid:latest -f docker/Dockerfile .
#
# Run (single container, same as host multi-process mode):
#   docker run --rm -v %cd%/data:/workspace/data \
#              -v %cd%/logs:/workspace/logs \
#              -v %cd%/checkpoints:/workspace/checkpoints \
#       pdc-hybrid:latest \
#       python src/hybrid/launcher.py --nworkers 2 \
#             --config configs/phase4_sA_hybrid.yaml
#
# For multi-container use, see docker/docker-compose.yml.

FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies first (layer cache-friendly)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source (data/ and logs/ are mounted as volumes at runtime)
COPY src/       ./src/
COPY configs/   ./configs/
COPY experiments/ ./experiments/

# Create runtime directories that may not be in the mounted volumes yet
RUN mkdir -p data logs checkpoints

# Default entrypoint: the hybrid launcher.
# Override CMD or pass full arguments with docker run.
ENTRYPOINT ["python", "src/hybrid/launcher.py"]
CMD ["--help"]
