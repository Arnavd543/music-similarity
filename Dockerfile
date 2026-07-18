# Pure-search API image: no GPU, no torch. api/main.py defers all heavy
# imports into the upload worker path, which runs on Modal, not here.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" qdrant-client numpy pandas scikit-learn pydantic \
    python-multipart

COPY pipeline/ pipeline/
COPY api/ api/
COPY training/ training/

# Set on the host: QDRANT_URL, QDRANT_API_KEY (secrets, not baked in)
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
