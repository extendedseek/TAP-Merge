# CPU reference image. Install a suitable PyTorch CUDA build for GPU experiments.
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir '.[hf,dev]'
CMD ["python", "-m", "tap_merge", "--help"]
