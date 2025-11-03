FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the code
COPY . .

# Expose port (Render needs this)
EXPOSE 10000

# Start the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
