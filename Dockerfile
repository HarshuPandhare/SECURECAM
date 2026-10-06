FROM python:3.11-slim

# ─── System dependencies required by OpenCV ───
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libxcb1 \
    libx11-6 \
    libx11-xcb1 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Install CPU-only PyTorch ───
# Azure App Service will run this container on CPU,
# so CUDA/NVIDIA packages are not required.
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# ─── Install application dependencies ───
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ─── Copy application ───
COPY app.py .

# ─── Copy frontend ───
COPY static/ ./static/

# ─── Copy trained YOLO model ───
COPY runs/detect/sixray_model/weights/best.pt \
    ./runs/detect/sixray_model/weights/best.pt

# ─── Copy training results ───
COPY runs/detect/sixray_model/results.csv \
    ./runs/detect/sixray_model/results.csv

# ─── Python configuration ───
ENV PYTHONUNBUFFERED=1

# ─── Application port ───
EXPOSE 8000

# ─── Start FastAPI/Uvicorn ───
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]