FROM python:3.11-slim

WORKDIR /yolov7

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install torch (GPU version; adjust if using CPU)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Copy YOLO repository code and install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the YOLO service code.
COPY . .

EXPOSE 8000

# Use Uvicorn to run the FastAPI app (assuming yolo_service.py defines "app")
ENTRYPOINT ["uvicorn", "yolo_service:app", "--host", "0.0.0.0", "--port", "8000"]