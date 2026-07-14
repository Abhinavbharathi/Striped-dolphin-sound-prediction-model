# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend ./backend

# Copy trained model
COPY models ./models

# Expose port (optional)
EXPOSE 8000

# Start FastAPI
CMD ["sh", "-c", "cd backend && uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
