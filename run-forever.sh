#!/bin/bash
# Keeps the bot running forever, restarting on crash after 5 seconds

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

while true; do
    echo "[$(date)] Starting bot..."
    python3 "$SCRIPT_DIR/bot.py"
    echo "[$(date)] Bot exited. Restarting in 5 seconds..."
    sleep 5
done
