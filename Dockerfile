# Use a slim Python image to reduce overall image size
FROM python:3.11-slim-bullseye

# Add labels for image metadata
LABEL maintainer="Suryansh Pandey" \
      version="1.0" \
      description="eBook Generator Application"

# Create a non-root user with a specific UID and GID
RUN addgroup --system --gid 1000 appuser && \
    adduser --system --uid 1000 --ingroup appuser appuser

# Set working directory
WORKDIR /app

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies as non-root user
USER appuser
RUN pip install --no-cache-dir --user --upgrade -r requirements.txt

# Copy the rest of the application
COPY --chown=appuser:appuser . .

# Set environment variables with default empty values
# These can be overridden at runtime
ENV GROQ_API_KEY=$GROQ_API_KEY \
    BLUESKY_USERNAME=$BLUESKY_USERNAME \
    BLUESKY_PASSWORD=$BLUESKY_PASSWORD \
    CLOUDINARY_CLOUD_NAME=$CLOUDINARY_CLOUD_NAME \
    CLOUDINARY_API_KEY=$CLOUDINARY_API_KEY \
    CLOUDINARY_API_SECRET=$CLOUDINARY_API_SECRET

# Use the non-root user
USER appuser

# Use a more explicit health check command if possible
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Set the default command
CMD ["python", "-u", "app.py"]