# Use a specific Python version to ensure consistency
FROM python:3.10.6-slim

# Update package repositories and install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-11-jdk-headless \
 && rm -rf /var/lib/apt/lists/*

# Expose the port on which your application will run
EXPOSE 8501

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install -r requirements.txt

# Copy the entire application source code and resources
COPY src/ ./src/
COPY resource/ ./resource/
COPY tests/ ./tests/

# Set environment variables
ENV PYSPARK_PYTHON=python3

# Command to run the application
CMD ["streamlit", "run", "src/app.py"]