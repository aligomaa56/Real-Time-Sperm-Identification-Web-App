# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install any needed packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for models if it doesn't exist
RUN mkdir -p app/static/models

# Download placeholder model file if it doesn't exist (to be replaced with actual model)
RUN if [ ! -f app/static/models/model.h5 ]; then \
    echo "Creating placeholder model file" && \
    touch app/static/models/model.h5; \
    fi

# Make port 5000 available
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app/app.py

# Run the application with Gunicorn
CMD ["gunicorn", "--worker-class", "eventlet", "--workers", "1", "--bind", "0.0.0.0:5000", "app.app:app"]