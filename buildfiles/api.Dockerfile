FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./

RUN pip install uv

RUN uv pip install --system -e .

COPY api/ ./api/
COPY models/ ./models/
COPY src/ ./src/

RUN mkdir -p /app/logs

RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["fastapi", "run", "api/main.py", "--host", "0.0.0.0", "--port", "8000"] 