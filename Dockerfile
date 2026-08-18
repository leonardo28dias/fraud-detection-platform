FROM python:3.10-slim

WORKDIR /app

# Install poetry
RUN pip install poetry==1.6.1

# Copy dependencies
COPY pyproject.toml poetry.lock* ./

# Install dependencies avoiding virtualenv for docker
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi

# Copy project files
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]