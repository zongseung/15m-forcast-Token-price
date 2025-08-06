# ───────────────────────────────────────────────────────────────────────────────
# Base: PyTorch with CUDA (이미 Python3 포함)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# ───────────────────────────────────────────────────────────────────────────────
# 시스템 패키지 설치 (cron만 추가)
RUN apt-get update \
 && apt-get install -y --no-install-recommends cron \
 && rm -rf /var/lib/apt/lists/*

# ───────────────────────────────────────────────────────────────────────────────
# 작업 디렉토리 및 PYTHONPATH 설정
WORKDIR /app/src
ENV PYTHONPATH=/app/src

# ───────────────────────────────────────────────────────────────────────────────
# 의존성 설치
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ───────────────────────────────────────────────────────────────────────────────
# 소스 복사
COPY src/ /app/src/

# ───────────────────────────────────────────────────────────────────────────────
# 크론 스크립트 및 등록 파일 복사
COPY train-cron /etc/cron.d/train-cron
COPY train-cron.sh /app/train-cron.sh
COPY forecast-cron.sh /app/forecast-cron.sh

# 권한 설정 및 크론 등록
RUN chmod +x /app/train-cron.sh /app/forecast-cron.sh \
 && chmod 0644 /etc/cron.d/train-cron \
 && crontab /etc/cron.d/train-cron
# ───────────────────────────────────────────────────────────────────────────────
# 로그 디렉터리 준비
RUN mkdir -p /var/log/cron && touch /var/log/cron.log

# ───────────────────────────────────────────────────────────────────────────────
# cron 포그라운드 실행
CMD ["cron", "-f"]
