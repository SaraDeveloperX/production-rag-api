# Base image
FROM python:3.9-slim

# Python settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Expose API port
EXPOSE 8000

# Run API (env vars passed at runtime)
# Supports $PORT for cloud platforms
CMD uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --proxy-headers \
    --forwarded-allow-ips='*'
