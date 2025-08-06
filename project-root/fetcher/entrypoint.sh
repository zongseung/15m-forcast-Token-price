#!/usr/bin/env bash
set -e

echo "▶️ Waiting for MySQL to be ready..."
# Bash built-in /dev/tcp 사용
while ! timeout 1 bash -c "</dev/tcp/db/3306"; do
  sleep 1
done
echo "▶️ MySQL is up - continuing"

while true; do
  python src/fetch_candles.py
  sleep 900
done

