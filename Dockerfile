FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies first (for better caching)
COPY requirements.txt .

# Install ffmpeg for audio processing (important for Whisper)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the code
COPY . .

# Expose Render's dynamic port (Render sets PORT automatically)
EXPOSE $PORT

# Start the FastAPI app using Render's PORT environment variable
CMD ["bash", "start.sh"]
