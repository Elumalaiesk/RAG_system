FROM python:3.11-slim

WORKDIR /app

# Install the CPU-only build of torch first. On Linux the default PyPI wheel
# bundles CUDA and pulls roughly 2.5 GB of GPU libraries that this service never
# uses - embeddings run on CPU. Pinning the CPU index cuts the image by an order
# of magnitude. Doing it before the main install means pip already has torch
# satisfied when sentence-transformers asks for it.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image. Without this the first request in
# every new container downloads ~90 MB from HuggingFace - slow, and a hard
# failure in an air-gapped or read-only deployment.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .
RUN mkdir -p storage/chroma storage/uploads

# The Chroma index and uploaded PDFs are state: mount a volume here, or every
# container restart starts with an empty index.
VOLUME ["/app/storage"]

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
