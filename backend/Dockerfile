FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY socbench ./socbench
RUN pip install --no-cache-dir . \
    && addgroup --system socbench \
    && adduser --system --ingroup socbench socbench

USER socbench

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=3)"]

CMD ["uvicorn", "socbench.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
