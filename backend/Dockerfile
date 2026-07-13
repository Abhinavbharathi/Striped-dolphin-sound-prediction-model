# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend files
COPY . .

# Expose Railway port (optional, Railway sets PORT automatically)
EXPOSE 8000

# Start FastAPI application
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
