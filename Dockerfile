# Vietnamese-RVC — Headless Automation API (GPU / CUDA 12.8, Blackwell RTX 50xx)
# Chạy: automation/headless_server.py  ->  http://0.0.0.0:8000
# LƯU Ý: RTX 5070 Ti = Blackwell (sm_120) BẮT BUỘC CUDA 12.8 + torch cu128.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# --- System deps: Python 3.10, ffmpeg, build tools ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        ffmpeg git curl build-essential \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

WORKDIR /app

# --- Python deps (được cache thành layer riêng để build lại nhanh) ---
# Pin numpy/numba rồi cài torch stack bản CUDA 12.8 (hỗ trợ Blackwell sm_120)
RUN pip install numpy==1.26.4 numba==0.61.0 && \
    pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128

# Các gói bootstrap (bản cross-platform, bỏ pywin32 vì chỉ dành cho Windows)
RUN pip install uv six packaging python-dateutil platformdirs onnxconverter_common wget

COPY requirements.txt .
RUN pip install -r requirements.txt

# Thư viện hệ thống cho audio (sounddevice cần libportaudio, soundfile cần libsndfile)
# Đặt sau lớp pip để không phá cache của các layer cài đặt nặng phía trên.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# --- Source code ---
COPY . .

# Thư mục dữ liệu (sẽ được mount volume trong docker-compose)
RUN mkdir -p assets audios dataset

EXPOSE 8000

CMD ["python", "automation/headless_server.py"]
