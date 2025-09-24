# LLM-Guided Classical ML - Reproducible Environment
FROM python:3.11.0-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables for reproducibility
ENV PYTHONHASHSEED=42
ENV PYTHONPATH=/app

# Create results directory
RUN mkdir -p results/data results/figures results/prompts_log

# Default command
CMD ["make", "reproduce"]

# Labels for metadata
LABEL maintainer="kojima.research@example.com"
LABEL description="LLM-Guided Learning for Classical Machine Learning"
LABEL version="1.0.0"
