FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY TITAN_env /app/titan_env
COPY server /app/server
COPY inference.py /app/inference.py
COPY openenv.yaml /app/openenv.yaml
COPY LICENSE /app/LICENSE

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["python", "-m", "titan_env.server.app"]
