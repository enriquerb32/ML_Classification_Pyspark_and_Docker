# Use a specific Python version to ensure consistency
FROM python:3.10.6-slim

# Update package repositories and install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-11-jdk-headless \
 && rm -rf /var/lib/apt/lists/*

# Expose the port on which your application will run
EXPOSE 8501

# Set the working directory and the container's virtual environment
RUN useradd --create-home enrique
USER enrique
WORKDIR /home/enrique

ENV VIRTUALENV=/home/enrique/venv
RUN python3 -m venv $VIRTUALENV
ENV PATH="$VIRTUALENV/bin:$PATH"

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application source code and resources
COPY src/ ./src/
COPY resource/ ./resource/
COPY tests/ ./tests/
COPY venv/ ./venv/
COPY pyproject.toml ./

# Set environment variables
ENV PYSPARK_PYTHON=python3

# Run tests to enhance an automated continuous integration
RUN python3 -m pytest tests/ && \
    python3 -m flake8 src/ && \
    python3 -m isort src/ --check && \
    python3 -m black src/ --check --quiet && \
    python3 -m pylint src/ --disable=C0114,C0116,R1705 && \
    python3 -m bandit -r src/ --quiet

# Command to run the application
CMD ["streamlit", "run", "src/app.py"]
