FROM python:3.12-slim

# Hugging Face Spaces runs containers as user 1000, so the app runs as that user everywhere:
# the same image behaves the same locally and in production.
RUN useradd --create-home --uid 1000 app
WORKDIR /app
RUN chown app:app /app
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    # Public deployment defaults: no sampled online judging (costs more than answering), and
    # request limits that cap LLM spend. Override with environment variables if needed.
    ONLINE_JUDGE_RATE=0 \
    ASK_LIMIT_PER_MINUTE=10 \
    ASK_LIMIT_PER_DAY=300 \
    LANGFUSE_ENV=production

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app config config
COPY --chown=app prompts prompts
COPY --chown=app data data
COPY --chown=app src src

USER app
# Build the index into the image: the container starts with its vectors and embedding models
# already on disk, needs no vector-DB server, and the image pins code + config + index together.
RUN python -m rag.index

EXPOSE 8000
CMD ["sh", "-c", "uvicorn rag.api:app --app-dir src --host 0.0.0.0 --port ${PORT:-8000}"]
