FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache
WORKDIR /app
RUN mkdir -p /app/.cache && chmod 777 /app/.cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libgomp1 git \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
COPY app.py .
EXPOSE 7860
CMD ["python","app.py"]
