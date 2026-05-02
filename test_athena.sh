#!/bin/bash
###############################################################################
# Athena — stop_athena.sh
# Stops all Athena services, kills GPU processes, frees all memory.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ATHENA_HOME="$SCRIPT_DIR"

echo "  Stopping Athena stack..."
echo "  ATHENA_HOME: $ATHENA_HOME"

# Stop compose stack
sudo docker compose --env-file "$ATHENA_HOME/.env" -f "$ATHENA_HOME/docker-compose.yml" down 2>/dev/null || true

# Force stop and remove containers
sudo docker stop athena-llm athena-whisper athena-tts athena-orchestrator 2>/dev/null || true
sudo docker rm athena-llm athena-whisper athena-tts athena-orchestrator 2>/dev/null || true

# Kill any leaked GPU processes
sudo fuser -k /dev/nvhost-ctrl 2>/dev/null || true
sudo fuser -k /dev/nvhost-gpu 2>/dev/null || true

# Aggressive memory clearing
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
sudo sh -c 'echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true

echo "  Done. All services stopped, GPU processes killed, memory freed."
echo "  Logs are in: $ATHENA_HOME/logs/"
