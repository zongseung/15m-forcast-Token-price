# fetcher/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 시스템 패키지 설치 (netcat)
RUN apt-get update && \
    apt-get install -y netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY entrypoint.sh ./

ENTRYPOINT ["bash", "entrypoint.sh"]
