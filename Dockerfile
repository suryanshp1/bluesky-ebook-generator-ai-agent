# Base image with Python
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file and bot script
COPY requirements.txt /app/
COPY app.py /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables (Replace these with real values during build or use Docker secrets)
ENV BLUESKY_USERNAME="your_bluesky_handle"
ENV BLUESKY_PASSWORD="your_password"
ENV OPENAI_API_KEY="your_openai_api_key"

# Run the bot script
CMD ["python", "app.py"]