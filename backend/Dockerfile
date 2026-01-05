# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and model files
COPY backend ./backend
COPY models ./models

# Expose Render's required port
EXPOSE 10000

# Start FastAPI server
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "10000"]
