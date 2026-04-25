FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY core/ core/
copy evolution/ evolution/
COPY infrastructure/ infrastructure/
COPY miners/ miners/
COPY web/ web/
COPY mine_until_ready.py .

RUN pip install --no-cache-dir .

RUN mkdir -p /app/cache /app/alpha /app/log /app/results

ENV PYTHONUNBUFFERED=1
ENV OLLAMA_URL=http://ollama:11434

EXPOSE 5000

CMD ["python", "-m", "core.alpha_orchestrator"]
