# Hugging Face Spaces (Docker SDK). Same Python as local development, so every pin in
# requirements.txt stays valid.
FROM python:3.14-slim

# HF Spaces runs the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# CPU-only PyTorch first: the default PyPI wheel bundles CUDA and is several GB larger.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Build the embedding index and download both models at build time, so the app starts
# in seconds and then loads models offline (see HF_HUB_OFFLINE in copilot/config.py).
RUN python -m scripts.build_index

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
