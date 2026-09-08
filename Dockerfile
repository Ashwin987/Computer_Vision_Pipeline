# Hugging Face Spaces Docker SDK - Streamlit is no longer a native Spaces SDK
# (confirmed against current HF docs: deprecated in favor of Docker SDK +
# Streamlit template), so this Dockerfile is the actual deployment mechanism,
# not an optional wrapper. See https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.11-slim

# libgl1/libglib2.0-0: headless OpenCV's real runtime shared-library needs
# (opencv-python-headless still links against these at import time).
# ffmpeg: moviepy's actual video encode/decode backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Non-root user pattern HF's own Docker Space docs specify (required for Dev
# Mode compatibility and to avoid permission issues on cache/write paths).
RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
# CPU-only torch build - the default PyPI wheel bundles CUDA binaries that
# are multi-GB and useless on cpu-basic hardware.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY --chown=user . /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]
