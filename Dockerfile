# Dockerfile: Containerize the spam detection system

# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the codebase
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command (inference mode)
CMD ["python", "predict.py", "This is a test message"]
