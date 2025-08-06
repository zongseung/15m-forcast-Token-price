#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
# .env 에서 TOKEN 등 불러오기
source /app/.env
python bot.py