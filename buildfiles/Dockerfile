FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

ENV UV_CACHE_DIR=/tmp/uv-cache
RUN mkdir -p $UV_CACHE_DIR

ADD . /app
WORKDIR /app

RUN uv sync --frozen && rm -rf $UV_CACHE_DIR

CMD ["uv", "run", "python", "main.py"]
