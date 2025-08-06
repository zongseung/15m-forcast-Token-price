#!/bin/bash
LOCKFILE="/tmp/train.lock"

if [ -e "$LOCKFILE" ]; then
    echo "train.py is already running"
    exit 1
fi

touch "$LOCKFILE"

cd /app/src
source /app/.env
export $(cat /app/.env | xargs)

/opt/conda/bin/python train.py >> /app/logs/train.log 2>&1

rm -f "$LOCKFILE"
