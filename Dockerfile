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
# CPU-only torch build - the default PyPI wheel bundles CUDA binaries that
# are multi-GB and useless on cpu-basic hardware. (requirements.txt's own
# first line already has this as a directive too - harmless to also pass it
# here explicitly.)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY --chown=user . /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]
