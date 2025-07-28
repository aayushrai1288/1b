# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies for tesseract and poppler
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y tesseract-ocr poppler-utils
# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY pipeline.py .

# Command to run the script
CMD ["python", "pipeline.py"]