FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create data directories
RUN mkdir -p data/fields data/operators data/shared_pool log

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "run_alpha_miner.py", "--llm", "deepseek", "--workers", "3"]
