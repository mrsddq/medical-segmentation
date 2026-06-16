FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PYTHONPATH=/app

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/outputs /app/runs \
    && chown -R appuser:appuser /app/outputs /app/runs
USER appuser

ENTRYPOINT ["python", "scripts/infer.py"]
