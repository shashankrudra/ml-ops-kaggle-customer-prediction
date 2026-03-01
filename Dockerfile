# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies
# libomp-dev is required for XGBoost multi-threading on Linux
RUN apt-get update && apt-get install -y \
    libomp-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Default command to run your script
CMD ["python", "src/train.py"]