# One image for the public demo: the built UI, the API, the demo recordings and
# the embedding model. Hugging Face Spaces runs it on port 7860.

# --- frontend -------------------------------------------------------------------
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- app ------------------------------------------------------------------------
FROM python:3.12-slim
# Spaces runs containers as uid 1000; the app writes its usage ledger under data/.
RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY backend/ backend/
# Editable install: paths (config/, recordings/, data/) resolve from /app.
RUN pip install --no-cache-dir -e .

COPY config/ config/
COPY recordings/demo/ recordings/demo/
COPY evals/ evals/
COPY --from=web /web/dist frontend/dist

# Bake the embedding model in, so the first workflow request doesn't download it.
# Not fatal: behind a restrictive network the model downloads on first use instead.
RUN (python -c "from tieout.config import get_settings; \
from tieout.retrieval.embeddings import get_embedder; \
get_embedder(get_settings().data_path / 'models').embed_query('warm up')" \
  || echo "embedding model not baked in; it will download on first use") \
 && mkdir -p data && chown -R user:user /app

USER user
# The demo recordings were made with this provider and model: request hashes
# include both, so changing them would turn every replay into a cache miss.
ENV DEMO_MODE=true \
    LLM_PROVIDER=gemini \
    LLM_MODEL=gemini-3.5-flash-lite \
    PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "tieout serve --host 0.0.0.0 --port ${PORT}"]
