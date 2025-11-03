FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies first (for better caching)
COPY requirements.txt .

# Install ffmpeg for audio processing (important for Whisper)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the application code
COPY . .

# Expose port (Render will use this to serve the API)
EXPOSE 10000

# Start the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
