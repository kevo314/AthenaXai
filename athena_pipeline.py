#!/bin/bash
###############################################################################
# Athena — test_athena.sh   (v55)
# Tests the 3 services that actually exist. Run while athena.sh is running.
#
# v55: removed the port-8000 LLM probe. The LLM has been merged into the
# orchestrator process since v43 — no separate llama-server listens on 8000.
###############################################################################

echo "============================================"
echo "  Athena Xai v1 — Service Tests"
echo "============================================"
echo ""

PASS=0
FAIL=0

# Test Whisper
echo "[1/3] Testing Whisper (whisper.cpp) on port 8001..."
if curl -sf --max-time 5 "http://localhost:8001/health" > /dev/null 2>&1; then
    echo "  -> Whisper is UP"
    PASS=$((PASS+1))
else
    echo "  -> Whisper is DOWN"
    FAIL=$((FAIL+1))
fi
echo ""

# Test TTS
echo "[2/3] Testing TTS on port 8002..."
if curl -sf --max-time 5 "http://localhost:8002/health" > /dev/null 2>&1; then
    echo "  -> TTS is UP"
    HEALTH=$(curl -s "http://localhost:8002/health")
    echo "  -> $HEALTH"
    PASS=$((PASS+1))
else
    echo "  -> TTS is DOWN"
    FAIL=$((FAIL+1))
fi
echo ""

# Test Orchestrator (in-process LLM lives here)
echo "[3/3] Testing Orchestrator (in-process LLM)..."
if sudo docker ps --format '{{.Names}}' | grep -q "athena-orchestrator"; then
    echo "  -> Orchestrator container is RUNNING"
    PASS=$((PASS+1))
else
    echo "  -> Orchestrator container is NOT running"
    FAIL=$((FAIL+1))
fi
echo ""

# Summary
echo "============================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "============================================"
if [ "$FAIL" -eq 0 ]; then
    echo "  All services operational."
    echo "  Speak into your microphone — Athena is listening."
else
    echo "  Some services failed. Check logs:"
    echo "    ls -la ~/athena/logs/"
    echo "    cat ~/athena/logs/whisper.log"
    echo "    cat ~/athena/logs/tts.log"
    echo "    cat ~/athena/logs/orchestrator.log"
    echo "    cat ~/athena/logs/pipeline.log"
fi
echo ""
