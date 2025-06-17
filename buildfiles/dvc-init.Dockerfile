FROM python:3.11-slim

# Install git and other dependencies needed for DVC
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy DVC requirements
COPY pyproject.toml ./
RUN pip install uv
RUN uv pip install --system dvc[s3]

# Copy Git repository and DVC configuration
COPY .git/ ./.git/
COPY dvc.yaml ./
COPY dvc.lock ./
COPY .dvc/ ./.dvc/

# Create the models directory
RUN mkdir -p /app/models

# Configure git (required by DVC)
RUN git config --global user.email "dvc@docker.container" && \
    git config --global user.name "DVC Container"

# Default command to pull models
CMD ["dvc", "pull", "models/catboost", "models/lgbm", "models/rf"] 