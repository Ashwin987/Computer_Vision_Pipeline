# Hugging Face Spaces Docker SDK - Streamlit is no longer a native Spaces SDK
# (confirmed against current HF docs: deprecated in favor of Docker SDK +
# Streamlit template), so this Dockerfile is the actual deployment mechanism,
# not an optional wrapper. See https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.11-slim

# No apt packages needed: confirmed during the Community Cloud deploy that
# opencv-python-headless is specifically built to not need libgl1/GTK/X11
# (that's the point of "headless"), and moviepy's imageio-ffmpeg dependency
# bundles its own portable ffmpeg binary via pip - no system ffmpeg required.

# Non-root user pattern HF's own Docker Space docs specify (required for Dev
# Mode compatibility and to avoid permission issues on cache/write paths).
RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user dashboard/requirements.txt requirements.txt
COPY --chown=user cv_pipeline/requirements.txt requirements-cv.txt
# Split across two files because Streamlit Community Cloud's default Python
# (3.14) has no compatible wheel for `inference`/onnxruntime, which broke
# that deployment's install entirely when both stacks lived in one file
# (pip/uv installs are all-or-nothing). This Dockerfile pins python:3.11-slim,
# where the CV stack installs fine, so both files are installed here to keep
# live upload working in a Docker Space.
RUN pip install --no-cache-dir -r requirements.txt -r requirements-cv.txt

COPY --chown=user . /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]
