#!/bin/bash
LOCKFILE="/tmp/forecast.lock"

if [ -e "$LOCKFILE" ]; then
    echo "forecast_scheduler.py is already running"
    exit 1
fi

touch "$LOCKFILE"

cd /app/src
source /app/.env
export $(cat /app/.env | xargs)

/opt/conda/bin/python forecast_scheduler.py >> /app/logs/forecast.log 2>&1

rm -f "$LOCKFILE"
