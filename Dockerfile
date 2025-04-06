# Stage 1: Build/Install Stage
FROM python:3.10-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /install_app

# Copy the requirements file
COPY requirements.txt .

# Install Python base and torch/torchvision from CPU index ONLY
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --prefix="/install" \
        torch==2.6.0 \
        torchvision==0.21.0

# Install the rest of the requirements
# Exclude torch/torchvision using grep to avoid reinstalling/conflict
# Use --extra-index-url so pip checks PyPI first, then the CPU index
# (Ensures Timm finds CPU torch without blocking finding Timm itself on PyPI)
RUN grep -vE '^torch|^torchvision' requirements.txt > requirements_other.txt && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu --prefix="/install" -r requirements_other.txt && \
    rm requirements_other.txt

# Stage 2: Final Runtime Stage
FROM python:3.10-slim AS final

# Install only necessary runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed dependencies from the builder stage's prefix dir
COPY --from=builder /install /usr/local

# Copy the application code and model into the container
COPY main.py .
COPY model/ /app/model/

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable for the port (optional, can be overridden)
ENV PORT=8080

# Command to run the application using uvicorn
# Use 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
