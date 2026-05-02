#!/bin/bash
###############################################################################
# Athena — Voice Preview
# Plays a test phrase in every available Kokoro voice at multiple speeds.
# Requires: the Athena stack running (athena-tts on port 8002)
#
# Usage:
#   ~/athena/preview_voices.sh
#   ~/athena/preview_voices.sh "Custom test phrase here"
###############################################################################

ATHENA_HOME="$(cd "$(dirname "$0")" && pwd)"
TTS_URL="http://localhost:8002"
TEST_PHRASE="${1:-Hello, I am Athena, your voice assistant. How can I help you today?}"
OUTPUT_DIR="$ATHENA_HOME/voice_samples"

# Colors for terminal
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}  Athena — Voice Preview${NC}"
echo -e "${BOLD}============================================${NC}"
echo ""

# Check TTS server is running
if ! curl -sf "$TTS_URL/health" >/dev/null 2>&1; then
    echo "ERROR: TTS server not responding at $TTS_URL"
    echo "Make sure the Athena stack is running first: ~/athena/athena.sh"
    exit 1
fi

# Get voice list from server
VOICES_JSON=$(curl -sf "$TTS_URL/voices" 2>/dev/null)
if [ -z "$VOICES_JSON" ]; then
    echo "ERROR: Could not fetch voice list from $TTS_URL/voices"
    exit 1
fi

# Parse voice names (simple grep approach, no jq dependency)
VOICES=$(echo "$VOICES_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for v in data.get('voices', []):
    print(v)
")

VOICE_COUNT=$(echo "$VOICES" | wc -l)
echo -e "  TTS Server: ${GREEN}online${NC}"
echo -e "  Voices:     ${VOICE_COUNT} available"
echo -e "  Phrase:     \"${TEST_PHRASE}\""
echo ""

mkdir -p "$OUTPUT_DIR"

# Speed options
SPEEDS=("0.8" "1.0" "1.2")

echo -e "${BOLD}Choose preview mode:${NC}"
echo "  1) Quick — all voices at speed 1.0"
echo "  2) Full  — all voices at speeds 0.8, 1.0, 1.2"
echo "  3) Pick  — choose a voice by number, try different speeds"
echo ""
read -p "Choice [1/2/3]: " MODE
echo ""

play_voice() {
    local voice=$1
    local speed=$2
    local outfile="$OUTPUT_DIR/${voice}_speed${speed}.wav"

    echo -ne "  Generating ${CYAN}${voice}${NC} @ speed ${speed}... "

    HTTP_CODE=$(curl -s -o "$outfile" -w "%{http_code}" \
        -X POST "$TTS_URL/synthesize" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$TEST_PHRASE\", \"voice\": \"$voice\", \"speed\": $speed}")

    if [ "$HTTP_CODE" = "200" ] && [ -f "$outfile" ] && [ -s "$outfile" ]; then
        DURATION=$(python3 -c "import soundfile; d=soundfile.info('$outfile'); print(f'{d.duration:.1f}s')" 2>/dev/null || echo "?s")
        echo -e "${GREEN}OK${NC} (${DURATION})"

        # Play through default audio device
        aplay -q "$outfile" 2>/dev/null || \
        paplay "$outfile" 2>/dev/null || \
        python3 -c "
import soundfile as sf, sounddevice as sd
data, sr = sf.read('$outfile')
sd.play(data, sr)
sd.wait()
" 2>/dev/null || echo "  (saved but could not play — check audio output)"
    else
        echo -e "${YELLOW}FAILED${NC} (HTTP $HTTP_CODE)"
    fi
}

case $MODE in
    1)
        echo -e "${BOLD}Playing all voices at speed 1.0:${NC}"
        echo ""
        IDX=0
        echo "$VOICES" | while read voice; do
            IDX=$((IDX + 1))
            echo -e "  ${BOLD}[$IDX/$VOICE_COUNT]${NC} ${CYAN}$voice${NC}"
            play_voice "$voice" "1.0"
            echo ""
            sleep 0.5
        done
        ;;
    2)
        echo -e "${BOLD}Playing all voices at speeds 0.8, 1.0, 1.2:${NC}"
        echo ""
        IDX=0
        echo "$VOICES" | while read voice; do
            IDX=$((IDX + 1))
            echo -e "  ${BOLD}[$IDX/$VOICE_COUNT]${NC} ${CYAN}$voice${NC}"
            for spd in "${SPEEDS[@]}"; do
                play_voice "$voice" "$spd"
            done
            echo ""
            sleep 0.5
        done
        ;;
    3)
        echo -e "${BOLD}Available voices:${NC}"
        IDX=0
        echo "$VOICES" | while read voice; do
            IDX=$((IDX + 1))
            echo "  $IDX) $voice"
        done
        echo ""

        while true; do
            read -p "Voice number (or 'q' to quit): " PICK
            [ "$PICK" = "q" ] && break

            VOICE=$(echo "$VOICES" | sed -n "${PICK}p")
            if [ -z "$VOICE" ]; then
                echo "  Invalid number. Try again."
                continue
            fi

            read -p "Speed [0.5-2.0, default 1.0]: " SPD
            SPD=${SPD:-1.0}

            echo ""
            play_voice "$VOICE" "$SPD"
            echo ""
        done
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac

echo ""
echo -e "${BOLD}============================================${NC}"
echo -e "  Voice samples saved to: ${OUTPUT_DIR}/"
echo -e "  To set your voice, edit the orchestrator config."
echo -e "${BOLD}============================================${NC}"
echo ""
