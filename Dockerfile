FROM python:3.12-slim

WORKDIR /app

# Install Poetry (locked version for reproducible builds)
RUN pip install --no-cache-dir poetry==1.8.5

# Copy dependency declarations first (better layer caching)
COPY pyproject.toml poetry.lock ./

# Install dependencies (main group only; tests are not needed at runtime)
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi

# Copy source code, configuration, and the API entrypoint script
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

EXPOSE 8000

# Model artifacts (models/*.pkl, metrics.json) are NOT baked into the image --
# they are gitignored and expected to be generated locally, then mounted as a
# volume at runtime (see docker-compose.yml).  The entrypoint script below
# verifies the artifact exists before starting the API.
CMD ["bash", "/app/scripts/start_api.sh"]
