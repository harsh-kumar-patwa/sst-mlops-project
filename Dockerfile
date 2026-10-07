FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config config
COPY prompts prompts
COPY data data
COPY src src

# Build the index into the image: the container starts with its vectors and embedding models
# already on disk, needs no vector-DB server, and the image tag pins code + config + index together.
RUN python -m rag.index

EXPOSE 8000
CMD ["sh", "-c", "uvicorn rag.api:app --app-dir src --host 0.0.0.0 --port ${PORT:-8000}"]
