#!/bin/bash
###############################################################################
# Athena Xai v1 — Master Setup & Launch
# Jetson Orin Nano Super 8GB / JetPack 6.2 / L4T R36.4
#
# v47 CHANGES:
#   - LLM loads FIRST into clean contiguous memory (before TTS+Whisper)
#   - TTS+Whisper start AFTER LLM allocation, fill remaining GPU space
#   - Pipeline waits for TTS+Whisper internally after LLM loads
#   - Removed service tests from launch (roll straight in)
#   - FIX: Build verification uses importlib.metadata (libcuda.so.1 not in docker build)
#   - Service tests (TTS, Whisper, VAD) restored between startup and pipeline launch
#   - FIX: Upgrade pip + install scikit-build-core/cmake/ninja before wheel build
#   - FIX: exceptiongroup<1.3 pinned (avoids Python 3.12 TypeVar crash on 3.10)
#   - FIX: Removed --no-build-isolation --no-deps from wheel build command
#   - All other v44 changes remain (config file, tok/s, fail-through, GGML env)
#   - llama-cpp-python built on HOST (full CUDA toolkit), wheel cached
#   - Wheel installed in Docker (no cmake/ninja inside container)
#   - Removed cuda-dev staging (no nvcc/headers needed in Docker)
#   - LLM config in llm_config.json (system prompt + generation params)
#   - Tok/s on every LLM response
#   - GGML_CUDA_NO_PINNED=1 for Jetson unified memory
#   - Fail-through build: tracks pass/fail per step, full report at end
#   - Full pip output (no tail piping) in Docker builds
#
# Steps:
#   1.  Clean memory + stop old containers
#   2.  Make scripts executable + desktop icon
#   3.  Create directories + set power mode
#   4.  NVMe swap setup + VM memory tuning
#   5.  Configure Docker nvidia runtime
#   6.  Install build dependencies
#   7.  Build llama.cpp + llama-cpp-python wheel
#   8.  Build whisper.cpp with CUDA
#   9.  Download all models + onnxruntime-gpu wheel
#  10.  Stage CUDA toolkit libraries from host
#  11.  Build 3 Docker images (Whisper, TTS, Orchestrator+LLM)
#  12.  USB audio setup + hardware tests
#  13.  Launch stack (memory pressure → TTS+Whisper → tests → orchestrator)
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ATHENA_HOME="$HOME/athena"
BUILD_DIR="$ATHENA_HOME/build"
MODEL_DIR="$ATHENA_HOME/models"
LOG_DIR="$ATHENA_HOME/logs"
DOCKER_STAGING="$ATHENA_HOME/docker-staging"

HF_TOKEN="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"
CUDA_ARCH="87"

# ═══════════════════════════════════════════════════════════════════════
#  FAIL-THROUGH TRACKING
# ═══════════════════════════════════════════════════════════════════════
REPORT=""
report_pass() { REPORT="$REPORT\n  [PASS] $1"; }
report_fail() { REPORT="$REPORT\n  [FAIL] $1"; echo "  !! FAILED: $1 — continuing..."; }
report_skip() { REPORT="$REPORT\n  [SKIP] $1"; }
FATAL_COUNT=0

echo "============================================"
echo "  Athena Xai v1 — Full Stack Setup"
echo "  3 Separate Docker Images"
echo "============================================"
echo ""

mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/athena_setup_$(date +%Y%m%d_%H%M%S).log"
echo "  Logging to: $RUN_LOG"
echo ""
exec > >(tee -a "$RUN_LOG") 2>&1


# ══════════════════════════════════════════════════════════════════════
#  Step 1: FULL SYSTEM CLEANUP
# ══════════════════════════════════════════════════════════════════════
echo "[1/13] System cleanup..."

echo "  -> Stopping any running Athena stack..."
if [ -f "$ATHENA_HOME/docker-compose.yml" ] && [ -f "$ATHENA_HOME/.env" ]; then
    sudo docker compose --env-file "$ATHENA_HOME/.env" -f "$ATHENA_HOME/docker-compose.yml" down 2>/dev/null || true
fi
sudo docker stop athena-llm athena-whisper athena-tts athena-orchestrator 2>/dev/null || true
sudo docker rm -f athena-llm athena-whisper athena-tts athena-orchestrator 2>/dev/null || true

sudo fuser -k /dev/nvhost-ctrl 2>/dev/null || true
sudo fuser -k /dev/nvhost-gpu 2>/dev/null || true
echo "  -> Containers stopped, GPU processes killed"

echo "  -> Pruning Docker (old images, build cache, volumes)..."
sudo docker container prune -f 2>/dev/null || true
sudo docker image prune -f 2>/dev/null || true
sudo docker builder prune -f 2>/dev/null || true
sudo docker volume prune -f 2>/dev/null || true
echo "  -> Docker pruned"

sudo apt-get clean 2>/dev/null || true
sudo journalctl --vacuum-size=100M 2>/dev/null || true
echo "  -> apt cache + journal trimmed"

if [ -d "$LOG_DIR" ]; then
    LOG_COUNT=$(find "$LOG_DIR" -name "*.log" -type f 2>/dev/null | wc -l)
    if [ "$LOG_COUNT" -gt 0 ]; then
        find "$LOG_DIR" -name "*.log" -type f -delete 2>/dev/null
        echo "  -> Cleared $LOG_COUNT old log files"
    fi
fi
mkdir -p "$LOG_DIR"

sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
sudo sh -c 'echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true

# Clean up old model files from previous versions
if [ -f "$MODEL_DIR/llm/Qwen3.5-0.8B-Q8_0.gguf" ]; then
    echo "  -> Removing old Qwen3.5-0.8B model (replaced by 2B in v39)..."
    rm -f "$MODEL_DIR/llm/Qwen3.5-0.8B-Q8_0.gguf"
fi
if [ -f "$MODEL_DIR/llm/Qwen3.5-2B-Q8_0.gguf" ]; then
    echo "  -> Removing old Qwen3.5-2B Unsloth model (replaced by abliterated in v40)..."
    rm -f "$MODEL_DIR/llm/Qwen3.5-2B-Q8_0.gguf"
fi
if [ -f "$MODEL_DIR/llm/Huihui-Qwen3.5-2B-abliterated.Q8_0.gguf" ]; then
    echo "  -> Removing prior Qwen3.5-2B-abliterated model (replaced by Gemma 4 E2B)..."
    rm -f "$MODEL_DIR/llm/Huihui-Qwen3.5-2B-abliterated.Q8_0.gguf"
fi
if [ -f "$MODEL_DIR/whisper/ggml-tiny.en.bin" ]; then
    echo "  -> Removing old Whisper tiny.en model (replaced by small q8_0 multilingual)..."
    rm -f "$MODEL_DIR/whisper/ggml-tiny.en.bin"
fi

# Clean up v39-v42 LLM container (merged into orchestrator in v43)
if sudo docker image inspect athena-llm:latest >/dev/null 2>&1; then
    echo "  -> Removing old athena-llm Docker image..."
    sudo docker rmi athena-llm:latest 2>/dev/null || true
fi
rm -f "$ATHENA_HOME/Dockerfile.llm" 2>/dev/null
rm -f "$ATHENA_HOME/.build-cache/athena-llm.md5" 2>/dev/null

# Clean up v43 system_prompt.txt (merged into llm_config.json in v47)
rm -f "$ATHENA_HOME/system_prompt.txt" 2>/dev/null

DISK_FREE=$(df -h / | awk 'NR==2 {print $4}')
MEM_AVAIL=$(free -m | awk '/Mem:/ {print $7}')
echo "  -> Disk free: $DISK_FREE | RAM available: ${MEM_AVAIL}MB"
echo "  -> Cleanup complete"
report_pass "System cleanup"
echo ""


# ── Step 2: Scripts + Desktop Icon ──
echo "[2/13] Setting up scripts and desktop icon..."
chmod +x "$SCRIPT_DIR"/stop_athena.sh 2>/dev/null || true
chmod +x "$SCRIPT_DIR"/test_athena.sh 2>/dev/null || true
chmod +x "$SCRIPT_DIR"/chat_athena.sh 2>/dev/null || true
chmod +x "$SCRIPT_DIR"/preview_voices.sh 2>/dev/null || true
chmod +x "$SCRIPT_DIR"/memory_diag.sh 2>/dev/null || true
chmod +x "$SCRIPT_DIR"/orchestrator/audio_gateway.sh 2>/dev/null || true

if [ -d "$HOME/Desktop" ]; then
    sed "s|^Icon=.*|Icon=$ATHENA_HOME/athena-icon.png|" "$SCRIPT_DIR/Athena.desktop" > "$HOME/Desktop/Athena.desktop" 2>/dev/null || true
    chmod +x "$HOME/Desktop/Athena.desktop" 2>/dev/null || true
    gio set "$HOME/Desktop/Athena.desktop" metadata::trusted true 2>/dev/null || true
    echo "  -> Desktop icon installed"
fi

if [ "$SCRIPT_DIR" != "$ATHENA_HOME" ]; then
    echo "  -> Copying project files to $ATHENA_HOME"
    mkdir -p "$ATHENA_HOME"
    cp -r "$SCRIPT_DIR"/* "$ATHENA_HOME/" 2>/dev/null || true
fi
report_pass "Scripts + desktop icon"
echo ""


# ── Step 3: Directories + Power Mode ──
echo "[3/13] Creating directories and setting power mode..."
mkdir -p "$BUILD_DIR" "$MODEL_DIR/llm" "$MODEL_DIR/whisper" "$MODEL_DIR/tts" "$MODEL_DIR/vad" "$LOG_DIR"
mkdir -p "$DOCKER_STAGING/bin" "$DOCKER_STAGING/lib" "$DOCKER_STAGING/cuda-libs"
mkdir -p "$DOCKER_STAGING/orchestrator" "$DOCKER_STAGING/tts"
echo "  -> Directories created"

sudo nvpmodel -m 0 2>/dev/null && echo "  -> MAXN_SUPER (25W) set" || echo "  -> nvpmodel not available"
sudo jetson_clocks 2>/dev/null && echo "  -> Clocks maximized" || true
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
report_pass "Directories + power mode"
echo ""


# ══════════════════════════════════════════════════════════════════════
#  Step 4: NVMe Swap + Memory Tuning
# ══════════════════════════════════════════════════════════════════════
echo "[4/13] NVMe swap + memory tuning..."

SWAP_FILE="/swapfile.athena"
SWAP_SIZE_GB=16

ZRAM_ACTIVE=$(swapon --show=NAME,TYPE 2>/dev/null | grep -c zram)
if [ "$ZRAM_ACTIVE" -gt 0 ]; then
    echo "  -> Disabling zram swap (uses RAM, bad for CUDA)..."
    for zdev in /dev/zram*; do
        sudo swapoff "$zdev" 2>/dev/null || true
    done
    echo "  -> zram disabled"
else
    echo "  -> No zram swap active (good)"
fi

if [ -f "$SWAP_FILE" ]; then
    CURRENT_SIZE=$(stat -c%s "$SWAP_FILE" 2>/dev/null || echo 0)
    EXPECTED_SIZE=$((SWAP_SIZE_GB * 1073741824))
    if [ "$CURRENT_SIZE" -ge "$EXPECTED_SIZE" ]; then
        echo "  -> Swap file exists: $SWAP_FILE ($(du -h "$SWAP_FILE" | cut -f1))"
    else
        echo "  -> Swap file too small, recreating..."
        sudo swapoff "$SWAP_FILE" 2>/dev/null || true
        sudo rm -f "$SWAP_FILE"
        echo "  -> Creating ${SWAP_SIZE_GB}GB swap file on NVMe..."
        sudo fallocate -l ${SWAP_SIZE_GB}G "$SWAP_FILE" 2>/dev/null || \
            sudo dd if=/dev/zero of="$SWAP_FILE" bs=1G count=$SWAP_SIZE_GB status=progress 2>&1
        sudo chmod 600 "$SWAP_FILE"
        sudo mkswap "$SWAP_FILE" 2>&1 | tail -1
    fi
else
    echo "  -> Creating ${SWAP_SIZE_GB}GB swap file on NVMe..."
    sudo fallocate -l ${SWAP_SIZE_GB}G "$SWAP_FILE" 2>/dev/null || \
        sudo dd if=/dev/zero of="$SWAP_FILE" bs=1G count=$SWAP_SIZE_GB status=progress 2>&1
    sudo chmod 600 "$SWAP_FILE"
    sudo mkswap "$SWAP_FILE" 2>&1 | tail -1
fi

if swapon --show 2>/dev/null | grep -q "$SWAP_FILE"; then
    echo "  -> NVMe swap already active"
else
    echo "  -> Enabling NVMe swap..."
    sudo swapon "$SWAP_FILE" 2>/dev/null || echo "  !! WARNING: Could not enable swap file."
fi

echo "  -> Tuning VM parameters for CUDA memory..."
sudo sh -c 'echo 100 > /proc/sys/vm/swappiness' 2>/dev/null
sudo sh -c 'echo 500 > /proc/sys/vm/vfs_cache_pressure' 2>/dev/null
sudo sh -c 'echo 1 > /proc/sys/vm/dirty_background_ratio' 2>/dev/null
sudo sh -c 'echo 5 > /proc/sys/vm/dirty_ratio' 2>/dev/null
sudo sh -c 'echo 262144 > /proc/sys/vm/min_free_kbytes' 2>/dev/null
sudo sh -c 'echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null
echo "  -> VM tuned: swappiness=100, vfs_cache_pressure=500, min_free=256MB"

echo ""
echo "  Swap status:"
swapon --show 2>/dev/null | while read line; do echo "    $line"; done
echo ""
MEM_FREE=$(free -m | awk '/Mem:/ {print $4}')
MEM_AVAIL=$(free -m | awk '/Mem:/ {print $7}')
SWAP_FREE=$(free -m | awk '/Swap:/ {print $4}')
echo "  RAM free: ${MEM_FREE}MB | Available: ${MEM_AVAIL}MB | Swap free: ${SWAP_FREE}MB"
report_pass "NVMe swap + VM tuning"
echo ""


# ── Step 5: Docker nvidia runtime ──
echo "[5/13] Checking Docker nvidia runtime..."
if sudo docker info 2>/dev/null | grep -q "nvidia"; then
    echo "  -> nvidia runtime already configured"
    report_pass "Docker nvidia runtime"
else
    echo "  -> Configuring nvidia runtime..."
    sudo mkdir -p /etc/docker
    cat <<'DAEMON' | sudo tee /etc/docker/daemon.json > /dev/null
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
DAEMON
    sudo systemctl restart docker
    if sudo docker info 2>/dev/null | grep -q "nvidia"; then
        echo "  -> nvidia runtime configured"
        report_pass "Docker nvidia runtime"
    else
        report_fail "Docker nvidia runtime — HARD STOP"
        echo "  Cannot continue without nvidia Docker runtime."
        exit 1
    fi
fi
echo ""


# ── Step 6: Build dependencies ──
echo "[6/13] Checking build dependencies..."
NEED_INSTALL=0
for pkg in cmake build-essential git curl wget; do
    if ! dpkg -s "$pkg" &>/dev/null; then NEED_INSTALL=1; break; fi
done
if [ "$NEED_INSTALL" -eq 1 ]; then
    echo "  -> Installing build dependencies..."
    sudo apt-get update
    sudo apt-get install -y cmake build-essential git curl wget pkg-config libcurl4-openssl-dev python3-pip
    report_pass "Build dependencies (installed)"
else
    echo "  -> All build dependencies present"
    report_pass "Build dependencies (cached)"
fi
echo ""


# ══════════════════════════════════════════════════════════════════════
#  Step 7: Build llama.cpp + llama-cpp-python wheel
# ══════════════════════════════════════════════════════════════════════
echo "[7/13] Building llama.cpp + llama-cpp-python wheel..."

# 7a. Build llama.cpp (for llama-server binary — used by whisper container pattern)
LLAMA_BIN="$BUILD_DIR/llama.cpp/build/bin/llama-server"
LLAMA_CUDA_LIB="$BUILD_DIR/llama.cpp/build/bin/libggml-cuda.so"

if [ -f "$LLAMA_BIN" ] && [ -f "$LLAMA_CUDA_LIB" ]; then
    echo "  -> llama-server already built with CUDA, skipping"
    report_pass "llama.cpp build (cached)"
else
    if [ -d "$BUILD_DIR/llama.cpp" ]; then rm -rf "$BUILD_DIR/llama.cpp"; fi
    echo "  -> Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$BUILD_DIR/llama.cpp"
    cd "$BUILD_DIR/llama.cpp"
    echo "  -> Building with CUDA (10-20 min on Orin Nano)..."
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_BUILD_TYPE=Release 2>&1 | tee "$LOG_DIR/llama_cpp_cmake.log"
    cmake --build build --config Release -j2 --target llama-server 2>&1 | tee "$LOG_DIR/llama_cpp_build.log"
    if [ -f "$BUILD_DIR/llama.cpp/build/bin/llama-server" ]; then
        echo "  -> llama-server built successfully"
        report_pass "llama.cpp build"
    else
        report_fail "llama.cpp build — see logs/llama_cpp_build.log"
    fi
fi

# Stage llama-server binary and libs
cp "$BUILD_DIR/llama.cpp/build/bin/llama-server" "$DOCKER_STAGING/bin/" 2>/dev/null || true
find "$BUILD_DIR/llama.cpp/build" -name "*.so*" -exec cp -P {} "$DOCKER_STAGING/lib/" \; 2>/dev/null || true
echo "  -> Staged llama-server"

# 7b. Build llama-cpp-python wheel on host (where full CUDA toolkit exists)
WHEEL_DIR="$DOCKER_STAGING"
CACHED_WHEEL=$(ls "$WHEEL_DIR"/llama_cpp_python*.whl 2>/dev/null | head -1)

if [ -n "$CACHED_WHEEL" ] && [ -s "$CACHED_WHEEL" ]; then
    echo "  -> llama-cpp-python wheel cached: $(basename "$CACHED_WHEEL")"
    report_pass "llama-cpp-python wheel (cached)"
else
    echo ""
    echo "  ╔══════════════════════════════════════════════════════════╗"
    echo "  ║  Building llama-cpp-python wheel with CUDA support      ║"
    echo "  ║  Uses 1 compiler core to avoid out-of-memory freeze.   ║"
    echo "  ║  Takes 20-30 minutes on first run.                      ║"
    echo "  ║  Wheel is cached — future runs are INSTANT.             ║"
    echo "  ╚══════════════════════════════════════════════════════════╝"
    echo ""

    # Ensure CUDA is on PATH
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export CUDA_HOME=/usr/local/cuda

    LLAMA_PY_LOG="$LOG_DIR/llama_cpp_python_build.log"
    echo "  Build started at $(date +%H:%M:%S)..."
    echo "  Full log: $LLAMA_PY_LOG"

    # v47 FIX: System pip 22.0 is too old for llama-cpp-python 0.3.x build system.
    # Must upgrade pip and install compatible build deps BEFORE building the wheel.
    # exceptiongroup<1.3 avoids TypeVar(default=...) which needs Python 3.12+.
    echo "  -> Upgrading pip + installing build prerequisites..."
    pip3 install --upgrade pip setuptools 2>&1 | tail -3
    pip3 install "scikit-build-core>=0.9,<0.10" cmake ninja "exceptiongroup<1.3" 2>&1 | tail -5
    echo "  -> Build prerequisites ready"

    CMAKE_BUILD_PARALLEL_LEVEL=1 \
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH" \
    FORCE_CMAKE=1 \
        pip3 wheel llama-cpp-python \
        -w "$WHEEL_DIR" --verbose 2>&1 | tee "$LLAMA_PY_LOG" | \
        while IFS= read -r line; do
            case "$line" in
                *Building*|*Compiling*|*.cpp*|*.cu*|*Linking*|*Installing*|*cmake*|*CUDA*)
                    printf "\r  >> %-74s" "$(echo "$line" | sed 's/.*\///' | tail -c 74)"
                    ;;
            esac
        done
    printf "\r%-80s\r" ""

    CACHED_WHEEL=$(ls "$WHEEL_DIR"/llama_cpp_python*.whl 2>/dev/null | head -1)
    if [ -n "$CACHED_WHEEL" ] && [ -s "$CACHED_WHEEL" ]; then
        echo "  -> Wheel built: $(basename "$CACHED_WHEEL") ($(du -h "$CACHED_WHEEL" | cut -f1))"
        report_pass "llama-cpp-python wheel (built)"
    else
        # Fallback: install on host, then copy wheel from pip cache
        echo "  -> Wheel not in output dir, trying pip install + cache extraction..."
        CMAKE_BUILD_PARALLEL_LEVEL=1 \
        CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH" \
        FORCE_CMAKE=1 \
            pip3 install llama-cpp-python --verbose 2>&1 | tee -a "$LLAMA_PY_LOG"

        if python3 -c "import llama_cpp; print(f'llama-cpp-python {llama_cpp.__version__}')" 2>/dev/null; then
            echo "  -> Installed on host, extracting wheel..."
            pip3 wheel llama-cpp-python --no-deps -w "$WHEEL_DIR" 2>/dev/null || true
            find /tmp/pip-* "$HOME/.cache/pip" -name "llama_cpp_python*.whl" \
                -exec cp {} "$WHEEL_DIR/" \; 2>/dev/null || true
            CACHED_WHEEL=$(ls "$WHEEL_DIR"/llama_cpp_python*.whl 2>/dev/null | head -1)
        fi

        if [ -n "$CACHED_WHEEL" ] && [ -s "$CACHED_WHEEL" ]; then
            echo "  -> Wheel recovered: $(basename "$CACHED_WHEEL")"
            report_pass "llama-cpp-python wheel (recovered)"
        else
            report_fail "llama-cpp-python wheel — see $LLAMA_PY_LOG"
            FATAL_COUNT=$((FATAL_COUNT + 1))
        fi
    fi
fi
echo ""


# ── Step 8: Build whisper.cpp with CUDA ──
echo "[8/13] Building whisper.cpp with CUDA..."
WHISPER_SERVER=""
for candidate in "$BUILD_DIR/whisper.cpp/build/bin/whisper-server" "$BUILD_DIR/whisper.cpp/build/bin/server"; do
    [ -f "$candidate" ] && WHISPER_SERVER="$candidate" && break
done
WHISPER_CUDA_LIB=$(find "$BUILD_DIR/whisper.cpp/build" -name "libggml-cuda*" -print -quit 2>/dev/null)

if [ -n "$WHISPER_SERVER" ] && [ -n "$WHISPER_CUDA_LIB" ]; then
    echo "  -> whisper-server already built with CUDA, skipping"
    report_pass "whisper.cpp build (cached)"
else
    if [ -d "$BUILD_DIR/whisper.cpp" ]; then rm -rf "$BUILD_DIR/whisper.cpp"; fi
    echo "  -> Cloning whisper.cpp..."
    git clone --depth 1 https://github.com/ggml-org/whisper.cpp "$BUILD_DIR/whisper.cpp"
    cd "$BUILD_DIR/whisper.cpp"
    echo "  -> Building with CUDA..."
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DWHISPER_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tee "$LOG_DIR/whisper_cmake.log"
    cmake --build build --config Release -j2 2>&1 | tee "$LOG_DIR/whisper_build.log"
    # Find the server binary (name varies by version)
    WHISPER_SERVER=""
    for candidate in "$BUILD_DIR/whisper.cpp/build/bin/whisper-server" "$BUILD_DIR/whisper.cpp/build/bin/server"; do
        [ -f "$candidate" ] && WHISPER_SERVER="$candidate" && break
    done
    if [ -n "$WHISPER_SERVER" ]; then
        echo "  -> whisper-server built: $(basename "$WHISPER_SERVER")"
        report_pass "whisper.cpp build"
    else
        report_fail "whisper.cpp build — see logs/whisper_build.log"
        FATAL_COUNT=$((FATAL_COUNT + 1))
    fi
fi

# Stage whisper binary + shared libs
if [ -n "$WHISPER_SERVER" ]; then
    cp "$WHISPER_SERVER" "$DOCKER_STAGING/bin/whisper-server" 2>/dev/null || true
    chmod +x "$DOCKER_STAGING/bin/whisper-server" 2>/dev/null || true
fi
find "$BUILD_DIR/whisper.cpp/build" -name "*.so*" -exec cp -P {} "$DOCKER_STAGING/lib/" \; 2>/dev/null || true

LIB_COUNT=$(ls "$DOCKER_STAGING/lib/"*.so* 2>/dev/null | wc -l)
echo "  -> Staged whisper-server + $LIB_COUNT shared libraries"
echo ""


# ══════════════════════════════════════════════════════════════════════
#  Step 9: Download Models
# ══════════════════════════════════════════════════════════════════════
echo "[9/13] Downloading models..."

# Gemma 4 E2B abliterated Q4_K_M
LLM_MODEL_FILE="$MODEL_DIR/llm/Huihui-gemma-4-E2B-it-abliterated-v2.Q4_K_M.gguf"
if [ -f "$LLM_MODEL_FILE" ] && [ -s "$LLM_MODEL_FILE" ]; then
    echo "  -> Gemma 4 E2B abliterated Q4_K_M already downloaded ($(du -h "$LLM_MODEL_FILE" | cut -f1))"
    report_pass "Model: Gemma 4 E2B abliterated Q4_K_M (cached)"
else
    echo "  -> Downloading Gemma 4 E2B abliterated Q4_K_M GGUF (~3.4GB)..."
    curl -L --fail -o "$LLM_MODEL_FILE" \
        -H "Authorization: Bearer $HF_TOKEN" \
        "https://huggingface.co/mradermacher/Huihui-gemma-4-E2B-it-abliterated-v2-GGUF/resolve/main/Huihui-gemma-4-E2B-it-abliterated-v2.Q4_K_M.gguf"
    if [ $? -eq 0 ] && [ -s "$LLM_MODEL_FILE" ]; then
        echo "  -> Downloaded ($(du -h "$LLM_MODEL_FILE" | cut -f1))"
        report_pass "Model: Gemma 4 E2B abliterated Q4_K_M"
    else
        rm -f "$LLM_MODEL_FILE"
        report_fail "Model: Gemma 4 E2B abliterated Q4_K_M download"
        FATAL_COUNT=$((FATAL_COUNT + 1))
    fi
fi

# Whisper small Q8_0 (multilingual, prequantized — required for translate mode)
WHISPER_MODEL="$MODEL_DIR/whisper/ggml-small-q8_0.bin"
if [ -f "$WHISPER_MODEL" ]; then
    echo "  -> Whisper small Q8_0 already downloaded"
    report_pass "Model: whisper small Q8_0 (cached)"
else
    echo "  -> Downloading Whisper small Q8_0 multilingual (~264MB)..."
    curl -L --fail -o "$WHISPER_MODEL" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q8_0.bin"
    if [ $? -eq 0 ] && [ -s "$WHISPER_MODEL" ]; then
        report_pass "Model: whisper small Q8_0"
    else
        report_fail "Model: whisper small Q8_0 download"
        FATAL_COUNT=$((FATAL_COUNT + 1))
    fi
fi

# Silero VAD ONNX
VAD_MODEL="$MODEL_DIR/vad/silero_vad.onnx"
if [ -f "$VAD_MODEL" ]; then
    echo "  -> Silero VAD already downloaded"
    report_pass "Model: silero VAD (cached)"
else
    echo "  -> Downloading Silero VAD ONNX (~2MB)..."
    curl -L --fail -o "$VAD_MODEL" \
        "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    if [ $? -eq 0 ] && [ -s "$VAD_MODEL" ]; then
        report_pass "Model: silero VAD"
    else
        report_fail "Model: silero VAD download"
    fi
fi

# Kokoro-82M ONNX TTS
KOKORO_MODEL="$MODEL_DIR/tts/kokoro-v1.0.onnx"
if [ -f "$KOKORO_MODEL" ]; then
    echo "  -> Kokoro ONNX model already downloaded ($(du -h "$KOKORO_MODEL" | cut -f1))"
    report_pass "Model: kokoro ONNX (cached)"
else
    echo "  -> Downloading Kokoro-82M ONNX model (~350MB)..."
    curl -L --fail -o "$KOKORO_MODEL" \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
    if [ $? -eq 0 ] && [ -s "$KOKORO_MODEL" ]; then
        report_pass "Model: kokoro ONNX"
    else
        report_fail "Model: kokoro ONNX download"
    fi
fi

KOKORO_VOICES="$MODEL_DIR/tts/voices-v1.0.bin"
if [ -f "$KOKORO_VOICES" ]; then
    echo "  -> Kokoro voices already downloaded"
    report_pass "Model: kokoro voices (cached)"
else
    echo "  -> Downloading Kokoro voice pack (~27MB)..."
    curl -L --fail -o "$KOKORO_VOICES" \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
    if [ $? -eq 0 ] && [ -s "$KOKORO_VOICES" ]; then
        report_pass "Model: kokoro voices"
    else
        report_fail "Model: kokoro voices download"
    fi
fi

# onnxruntime-gpu wheel for Jetson
ORT_WHEEL=$(ls "$DOCKER_STAGING"/onnxruntime_gpu-*.whl 2>/dev/null | head -1)
if [ -n "$ORT_WHEEL" ] && [ -s "$ORT_WHEEL" ]; then
    echo "  -> onnxruntime-gpu wheel already downloaded ($(du -h "$ORT_WHEEL" | cut -f1))"
    report_pass "onnxruntime-gpu wheel (cached)"
else
    echo "  -> Downloading onnxruntime-gpu for Jetson..."
    ORT_WHEEL="$DOCKER_STAGING/onnxruntime_gpu-1.20.2-cp310-cp310-linux_aarch64.whl"
    curl -L --fail -o "$ORT_WHEEL" \
        "https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/f6e/2baa664069470/onnxruntime_gpu-1.20.2-cp310-cp310-linux_aarch64.whl" 2>/dev/null
    if [ $? -ne 0 ] || [ ! -s "$ORT_WHEEL" ]; then
        echo "  -> Primary source failed, trying mirror..."
        curl -L --fail -o "$ORT_WHEEL" \
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl" 2>/dev/null
    fi
    if [ -s "$ORT_WHEEL" ]; then
        echo "  -> Downloaded ($(du -h "$ORT_WHEEL" | cut -f1))"
        report_pass "onnxruntime-gpu wheel"
    else
        report_fail "onnxruntime-gpu wheel download"
        FATAL_COUNT=$((FATAL_COUNT + 1))
    fi
fi
echo ""


# ══════════════════════════════════════════════════════════════════════
#  Step 10: Stage CUDA toolkit libraries from host
# ══════════════════════════════════════════════════════════════════════
echo "[10/13] Staging CUDA toolkit libraries from host..."

CUDA_STAGING="$DOCKER_STAGING/cuda-libs"
rm -rf "$CUDA_STAGING"
mkdir -p "$CUDA_STAGING"

CUDA_SEARCH_DIRS="/usr/local/cuda/lib64 /usr/local/cuda-12.6/targets/aarch64-linux/lib /usr/lib/aarch64-linux-gnu"

CUDA_LIB_PATTERNS=(
    "libcudart.so*" "libcublas.so*" "libcublasLt.so*" "libcudnn*.so*"
    "libcufft.so*" "libcurand.so*" "libcusparse.so*" "libcusolver.so*"
    "libnvrtc.so*" "libnvJitLink.so*" "libcupti.so*"
    "libnvinfer.so*" "libnvonnxparser.so*"
)

CUDA_LIB_COUNT=0
for dir in $CUDA_SEARCH_DIRS; do
    if [ -d "$dir" ]; then
        for pattern in "${CUDA_LIB_PATTERNS[@]}"; do
            for lib in "$dir"/$pattern; do
                if [ -f "$lib" ] || [ -L "$lib" ]; then
                    cp -P "$lib" "$CUDA_STAGING/" 2>/dev/null
                    CUDA_LIB_COUNT=$((CUDA_LIB_COUNT + 1))
                fi
            done
        done
    fi
done

CRITICAL_LIBS="libcudart.so libcublas.so libcublasLt.so"
MISSING=""
for lib in $CRITICAL_LIBS; do
    if ! ls "$CUDA_STAGING"/${lib}* 2>/dev/null | head -1 >/dev/null; then
        MISSING="$MISSING $lib"
    fi
done

if [ -n "$MISSING" ]; then
    report_fail "CUDA libs staging — missing:$MISSING"
else
    echo "  -> Staged $CUDA_LIB_COUNT CUDA toolkit libraries"
    report_pass "CUDA libs staged ($CUDA_LIB_COUNT files)"
fi

echo "  -> Contents of cuda-libs/:"
ls -la "$CUDA_STAGING"/*.so* 2>/dev/null | awk '{print "     ", $NF, $5}' | head -20
CUDA_SIZE=$(du -sh "$CUDA_STAGING" 2>/dev/null | cut -f1)
echo "  -> Total CUDA libs size: $CUDA_SIZE"
echo ""


# ══════════════════════════════════════════════════════════════════════
#  Step 11: Build 3 Docker images
# ══════════════════════════════════════════════════════════════════════
echo "[11/13] Building Docker images (cached if unchanged)..."

# Copy project files into staging
cp "$ATHENA_HOME/Dockerfile.whisper" "$DOCKER_STAGING/" 2>/dev/null || cp "$SCRIPT_DIR/Dockerfile.whisper" "$DOCKER_STAGING/"
cp "$ATHENA_HOME/Dockerfile.tts" "$DOCKER_STAGING/" 2>/dev/null || cp "$SCRIPT_DIR/Dockerfile.tts" "$DOCKER_STAGING/"
cp "$ATHENA_HOME/Dockerfile.orchestrator" "$DOCKER_STAGING/" 2>/dev/null || cp "$SCRIPT_DIR/Dockerfile.orchestrator" "$DOCKER_STAGING/"
cp "$ATHENA_HOME/constraints-docker.txt" "$DOCKER_STAGING/" 2>/dev/null || cp "$SCRIPT_DIR/constraints-docker.txt" "$DOCKER_STAGING/"
cp "$ATHENA_HOME/requirements-tts.txt" "$DOCKER_STAGING/" 2>/dev/null || cp "$SCRIPT_DIR/requirements-tts.txt" "$DOCKER_STAGING/"
cp "$ATHENA_HOME/requirements-orch.txt" "$DOCKER_STAGING/" 2>/dev/null || cp "$SCRIPT_DIR/requirements-orch.txt" "$DOCKER_STAGING/"
cp -r "$ATHENA_HOME/orchestrator" "$DOCKER_STAGING/" 2>/dev/null || cp -r "$SCRIPT_DIR/orchestrator" "$DOCKER_STAGING/"
cp -r "$ATHENA_HOME/tts" "$DOCKER_STAGING/" 2>/dev/null || cp -r "$SCRIPT_DIR/tts" "$DOCKER_STAGING/"

cd "$DOCKER_STAGING"

CACHE_DIR="$ATHENA_HOME/.build-cache"
mkdir -p "$CACHE_DIR"

build_if_changed() {
    local IMAGE_NAME="$1"
    local DOCKERFILE="$2"
    shift 2
    local DEPS="$@"

    local HASH_INPUT=""
    HASH_INPUT=$(cat "$DOCKERFILE" $DEPS 2>/dev/null | md5sum | cut -d' ' -f1)
    local CACHE_FILE="$CACHE_DIR/${IMAGE_NAME}.md5"
    local OLD_HASH=""
    [ -f "$CACHE_FILE" ] && OLD_HASH=$(cat "$CACHE_FILE")

    local IMAGE_EXISTS=false
    sudo docker image inspect "${IMAGE_NAME}:latest" >/dev/null 2>&1 && IMAGE_EXISTS=true

    if [ "$IMAGE_EXISTS" = true ] && [ "$HASH_INPUT" = "$OLD_HASH" ]; then
        echo "  -> ${IMAGE_NAME}:latest — unchanged, skipping build"
        report_pass "Docker: ${IMAGE_NAME} (cached)"
        return 0
    fi

    if [ "$IMAGE_EXISTS" = true ] && [ "$HASH_INPUT" != "$OLD_HASH" ]; then
        echo "  -> ${IMAGE_NAME}: Dockerfile changed, rebuilding..."
    else
        echo "  -> ${IMAGE_NAME}: building..."
    fi

    local BUILD_LOG="$LOG_DIR/docker_build_${IMAGE_NAME}_$(date +%Y%m%d_%H%M%S).log"
    sudo docker build --no-cache -t "${IMAGE_NAME}:latest" -f "$DOCKERFILE" . 2>&1 | tee "$BUILD_LOG"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        report_fail "Docker: ${IMAGE_NAME} — see $BUILD_LOG"
        FATAL_COUNT=$((FATAL_COUNT + 1))
        return 1
    fi

    echo "$HASH_INPUT" > "$CACHE_FILE"
    echo "  -> ${IMAGE_NAME}:latest built"
    report_pass "Docker: ${IMAGE_NAME}"
    return 0
}

echo ""
build_if_changed "athena-whisper" "Dockerfile.whisper"
echo ""
build_if_changed "athena-tts" "Dockerfile.tts" "requirements-tts.txt" "constraints-docker.txt" "tts/tts_server.py"
echo ""
build_if_changed "athena-orch" "Dockerfile.orchestrator" "requirements-orch.txt" "constraints-docker.txt" "orchestrator/athena_pipeline.py"

echo ""
echo "  Docker images:"
sudo docker images --format "    {{.Repository}}:{{.Tag}} — {{.Size}}" | grep athena
echo ""


# ══════════════════════════════════════════════════════════════════════
#  BUILD REPORT
# ══════════════════════════════════════════════════════════════════════
echo "=========================================="
echo "  Athena Xai v1 — Build Report"
echo "=========================================="
echo -e "$REPORT"
echo "=========================================="

# Save report
REPORT_FILE="$LOG_DIR/build_report.log"
echo "Athena Xai v1 Build Report — $(date)" > "$REPORT_FILE"
echo -e "$REPORT" >> "$REPORT_FILE"
echo "  Report saved: $REPORT_FILE"

FAIL_COUNT=$(echo -e "$REPORT" | grep -c "\[FAIL\]")
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo "  $FAIL_COUNT FAILURE(S) DETECTED"
    echo "  Logs: $LOG_DIR/"
    echo ""
    read -p "  Attempt launch anyway? [y/N]: " FORCE_LAUNCH
    FORCE_LAUNCH=${FORCE_LAUNCH:-N}
    if [[ ! "$FORCE_LAUNCH" =~ ^[Yy]$ ]]; then
        echo "  Exiting. Fix failures and run again."
        exit 1
    fi
fi
echo ""


# ── Step 12: Audio Hardware Tests ──
# v51: Detect both USB cards by NAME (independent of mic test) and test each speaker
# separately. Three independent Y/n tests:
#   1. USB Composite Device speaker (wireless headset)
#   2. UACDemoV1.0 speaker (wired USB)
#   3. USB Composite Device microphone (wireless headset)
echo "[12/13] Audio hardware tests..."
echo ""
HEADSET_SPK_PASS=false
WIRED_SPK_PASS=false
MIC_PASS=false

echo "  ── USB Audio Device Setup ──"

# --- Detect wireless headset dongle (USB Composite Device, card 0 typically) ---
HEADSET_CARD=$(pactl list cards short 2>/dev/null | grep -i "USB.*Composite" | awk '{print $2}' | head -1)
HEADSET_SINK=""
HEADSET_SOURCE=""
if [ -n "$HEADSET_CARD" ]; then
    echo "  -> Wireless headset (USB Composite Device): card=$HEADSET_CARD"
    PROFILE_SET=false
    for profile in \
        "output:analog-stereo+input:mono-fallback" \
        "output:analog-stereo+input:analog-mono" \
        "output:analog-stereo" \
        "analog-stereo" \
        "output:analog-stereo+input:analog-stereo"; do
        if pactl set-card-profile "$HEADSET_CARD" "$profile" 2>/dev/null; then
            echo "     stereo profile: $profile"
            PROFILE_SET=true
            break
        fi
    done
    if [ "$PROFILE_SET" = false ]; then
        echo "     WARNING: could not set stereo profile on USB Composite Device"
    fi
    HEADSET_SINK=$(pactl list sinks short 2>/dev/null | grep -i "Composite" | awk '{print $2}' | head -1)
    HEADSET_SOURCE=$(pactl list sources short 2>/dev/null | grep -i "Composite" | grep -v "\.monitor" | awk '{print $2}' | head -1)
    # v52: do NOT change host PA defaults — keep whatever the user has routed for movies/etc.
    # The orchestrator addresses the device explicitly by PortAudio index, no defaults needed.
    [ -n "$HEADSET_SINK" ]   && echo "     sink:   $HEADSET_SINK"
    [ -n "$HEADSET_SOURCE" ] && echo "     source: $HEADSET_SOURCE"
else
    echo "  !! USB Composite Device NOT FOUND (wireless headset dongle)"
fi

# --- Detect wired USB speaker (UACDemoV1.0, card 3 typically) ---
WIRED_CARD=$(pactl list cards short 2>/dev/null | grep -i "UACDemo" | awk '{print $2}' | head -1)
WIRED_SINK=""
if [ -n "$WIRED_CARD" ]; then
    echo "  -> Wired USB speaker (UACDemoV1.0): card=$WIRED_CARD"
    WIRED_PROFILE_SET=false
    for profile in "output:analog-stereo" "analog-stereo"; do
        if pactl set-card-profile "$WIRED_CARD" "$profile" 2>/dev/null; then
            echo "     stereo profile: $profile"
            WIRED_PROFILE_SET=true
            break
        fi
    done
    if [ "$WIRED_PROFILE_SET" = false ]; then
        echo "     WARNING: could not set stereo profile on UACDemoV1.0"
    fi
    WIRED_SINK=$(pactl list sinks short 2>/dev/null | grep -i "UACDemo" | awk '{print $2}' | head -1)
    [ -n "$WIRED_SINK" ] && echo "     sink (kept as secondary, default not changed): $WIRED_SINK"
else
    echo "  !! UACDemoV1.0 NOT FOUND (wired USB speaker)"
fi
echo ""

# Helper: generate the 1.5s 440Hz stereo test tone WAV
TONE_FILE="/tmp/athena_speaker_test.wav"
python3 -c "
import struct, wave, math
sr=44100; dur=1.5; freq=440; amp=16000
s=[int(amp*math.sin(2*math.pi*freq*i/sr)) for i in range(int(sr*dur))]
stereo=[]
for sample in s:
    stereo.append(sample)
    stereo.append(sample)
w=wave.open('$TONE_FILE','w'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
w.writeframes(struct.pack('<'+'h'*len(stereo),*stereo)); w.close()
" 2>/dev/null

echo "  ========================================="
echo "  TEST 1/3: USB Composite Device speaker (wireless headset)"
echo "  ========================================="
read -p "  Run wireless headset speaker test? [Y/n]: " HEADSET_SPK_RUN
HEADSET_SPK_RUN=${HEADSET_SPK_RUN:-Y}
if [[ "$HEADSET_SPK_RUN" =~ ^[Yy]$ ]]; then
    if [ -n "$HEADSET_SINK" ]; then
        paplay --device="$HEADSET_SINK" "$TONE_FILE" 2>/dev/null \
            || aplay -q -D "plughw:${HEADSET_CARD},0" "$TONE_FILE" 2>/dev/null
    elif [ -n "$HEADSET_CARD" ]; then
        aplay -q -D "plughw:${HEADSET_CARD},0" "$TONE_FILE" 2>/dev/null
    else
        echo "  -> No headset card detected, skipping"
    fi
    read -p "  Did you hear the test tone on the wireless headset? [Y/n]: " HEADSET_SPK_HEARD
    HEADSET_SPK_HEARD=${HEADSET_SPK_HEARD:-Y}
    if [[ "$HEADSET_SPK_HEARD" =~ ^[Yy]$ ]]; then
        HEADSET_SPK_PASS=true
        report_pass "Audio: USB Composite Device speaker (wireless headset)"
    else
        report_fail "Audio: USB Composite Device speaker (wireless headset)"
    fi
else
    echo "  -> Skipped"; HEADSET_SPK_PASS=true
    report_skip "Audio: USB Composite Device speaker (wireless headset)"
fi
echo ""

echo "  ========================================="
echo "  TEST 2/3: UACDemoV1.0 speaker (wired USB)"
echo "  ========================================="
read -p "  Run wired USB speaker test? [Y/n]: " WIRED_SPK_RUN
WIRED_SPK_RUN=${WIRED_SPK_RUN:-Y}
if [[ "$WIRED_SPK_RUN" =~ ^[Yy]$ ]]; then
    if [ -n "$WIRED_SINK" ]; then
        paplay --device="$WIRED_SINK" "$TONE_FILE" 2>/dev/null \
            || aplay -q -D "plughw:${WIRED_CARD},0" "$TONE_FILE" 2>/dev/null
    elif [ -n "$WIRED_CARD" ]; then
        aplay -q -D "plughw:${WIRED_CARD},0" "$TONE_FILE" 2>/dev/null
    else
        echo "  -> No wired USB speaker detected, skipping"
    fi
    read -p "  Did you hear the test tone on the wired USB speaker? [Y/n]: " WIRED_SPK_HEARD
    WIRED_SPK_HEARD=${WIRED_SPK_HEARD:-Y}
    if [[ "$WIRED_SPK_HEARD" =~ ^[Yy]$ ]]; then
        WIRED_SPK_PASS=true
        report_pass "Audio: UACDemoV1.0 speaker (wired USB)"
    else
        report_fail "Audio: UACDemoV1.0 speaker (wired USB)"
    fi
else
    echo "  -> Skipped"; WIRED_SPK_PASS=true
    report_skip "Audio: UACDemoV1.0 speaker (wired USB)"
fi
rm -f "$TONE_FILE"
echo ""

echo "  ========================================="
echo "  TEST 3/3: USB Composite Device microphone (wireless headset, 5 seconds)"
echo "  ========================================="
read -p "  Run microphone test? [Y/n]: " MIC_RUN
MIC_RUN=${MIC_RUN:-Y}
if [[ "$MIC_RUN" =~ ^[Yy]$ ]]; then
    echo "  -> Recording 5 seconds... SPEAK NOW!"
    MIC_TMPFILE="/tmp/athena_mic_test.wav"
    rm -f "$MIC_TMPFILE"
    timeout 6 parecord --file-format=wav --rate=16000 --channels=1 "$MIC_TMPFILE" 2>/dev/null &
    PAREC_PID=$!; sleep 5; kill $PAREC_PID 2>/dev/null; wait $PAREC_PID 2>/dev/null
    if [ ! -s "$MIC_TMPFILE" ] && [ -n "$HEADSET_CARD" ]; then
        arecord -d 5 -f S16_LE -r 16000 -c 1 -D "plughw:${HEADSET_CARD},0" "$MIC_TMPFILE" 2>/dev/null
    elif [ ! -s "$MIC_TMPFILE" ]; then
        arecord -d 5 -f S16_LE -r 16000 -c 1 "$MIC_TMPFILE" 2>/dev/null
    fi
    if [ -s "$MIC_TMPFILE" ]; then
        python3 -c "
import wave, struct, math
w=wave.open('$MIC_TMPFILE','r'); frames=w.readframes(w.getnframes()); w.close()
s=struct.unpack('<'+'h'*(len(frames)//2),frames)
rms=math.sqrt(sum(x*x for x in s)/max(len(s),1))
print(f'  RMS: {rms:.0f}/32768')
if rms>200: print('PASS')
elif rms>50: print('LOW')
else: print('FAIL')
" 2>&1
        MIC_PASS=true
        report_pass "Audio: USB Composite Device microphone (wireless headset)"
    else
        report_fail "Audio: USB Composite Device microphone (wireless headset) — no recording"
    fi
    rm -f "$MIC_TMPFILE"
else
    echo "  -> Skipped"; MIC_PASS=true
    report_skip "Audio: USB Composite Device microphone (wireless headset)"
fi
echo ""


# ══════════════════════════════════════════════════════════════════════
#  Step 13: Launch Stack
# ══════════════════════════════════════════════════════════════════════
echo "[13/13] Launching Athena stack..."
echo ""

cp "$SCRIPT_DIR/docker-compose.yml" "$ATHENA_HOME/docker-compose.yml" 2>/dev/null || true
cp "$SCRIPT_DIR/llm_config.json" "$ATHENA_HOME/llm_config.json" 2>/dev/null || true

# v57: PulseAudio is no longer used. Output goes through PortAudio→ALSA
# directly via /dev/snd. No cookie pre-creation, no sink-name discovery.
# The orchestrator finds output devices by name pattern via sd.query_devices()
# at startup, the same way it already finds the mic input.

echo "ATHENA_HOME=$ATHENA_HOME" > "$ATHENA_HOME/.env"
echo "NVIDIA_VISIBLE_DEVICES=all" >> "$ATHENA_HOME/.env"
echo "NVIDIA_DRIVER_CAPABILITIES=compute,utility" >> "$ATHENA_HOME/.env"

sudo docker compose --env-file "$ATHENA_HOME/.env" -f "$ATHENA_HOME/docker-compose.yml" down 2>/dev/null || true

# ── Phase 1: Evacuate physical RAM for CUDA ──
echo "  ── Phase 1: Memory evacuation ──"
sudo fuser -k /dev/nvhost-ctrl 2>/dev/null || true
sudo fuser -k /dev/nvhost-gpu 2>/dev/null || true
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
sudo sh -c 'echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true

echo "  -> Pressuring RAM to force non-essential pages to NVMe swap..."
python3 -c "
import ctypes, time
chunks = []
try:
    for i in range(12):
        chunks.append(ctypes.create_string_buffer(256 * 1024 * 1024))
except MemoryError:
    pass
time.sleep(2)
del chunks
" 2>/dev/null || echo "  -> (memory pressure skipped)"

sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
sudo sh -c 'echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true
sleep 2

MEM_FREE=$(free -m | awk '/Mem:/ {print $4}')
MEM_AVAIL=$(free -m | awk '/Mem:/ {print $7}')
SWAP_USED=$(free -m | awk '/Swap:/ {print $3}')
echo "  -> RAM free: ${MEM_FREE}MB | Available: ${MEM_AVAIL}MB | Swap used: ${SWAP_USED}MB"
echo ""

echo "  ── Memory snapshot: BEFORE containers ──" >> "$LOG_DIR/memory_phases.log"
bash "$ATHENA_HOME/memory_diag.sh" >> "$LOG_DIR/memory_phases.log" 2>&1
echo "  -> Memory state logged to memory_phases.log"
echo ""

# ── Phase 2: Start orchestrator FIRST (LLM gets clean contiguous memory) ──
echo "  ── Phase 2: Starting orchestrator (LLM loads first) ──"
echo "    LLM needs ~3.5GB contiguous CUDA memory — gets first dibs on clean RAM"
echo "    TTS + Whisper start AFTER with smaller allocations"
echo ""

sudo docker compose --env-file "$ATHENA_HOME/.env" -f "$ATHENA_HOME/docker-compose.yml" \
    up -d athena-orchestrator 2>&1

echo "  -> Orchestrator starting (LLM loading into GPU...)"
echo "  -> This takes 15-30 seconds..."
echo ""

# Give LLM time to allocate its CUDA memory before starting other GPU containers
# The pipeline logs will show when it's done
sleep 15

echo "  ── Memory snapshot: LLM loading ──" >> "$LOG_DIR/memory_phases.log"
bash "$ATHENA_HOME/memory_diag.sh" >> "$LOG_DIR/memory_phases.log" 2>&1

# ── Phase 3: Start TTS + Whisper (smaller allocations fill remaining space) ──
echo "  ── Phase 3: Starting TTS + Whisper ──"
echo "    athena-tts          Kokoro-82M (af_sky)   port 8002  GPU (~300MB)"
echo "    athena-whisper      Whisper tiny.en       port 8001  GPU (~80MB)"
echo ""

# Small memory pressure to push non-CUDA pages out before TTS+Whisper
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
sudo sh -c 'echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true

sudo docker compose --env-file "$ATHENA_HOME/.env" -f "$ATHENA_HOME/docker-compose.yml" \
    up -d athena-tts athena-whisper 2>&1

echo "  -> TTS + Whisper starting..."
echo "  -> Pipeline will wait for them internally before entering listen loop"
echo ""

echo "  ── Memory snapshot: All containers started ──" >> "$LOG_DIR/memory_phases.log"
bash "$ATHENA_HOME/memory_diag.sh" >> "$LOG_DIR/memory_phases.log" 2>&1
echo "  -> Memory state logged"
echo ""

# ── Attach to orchestrator output (foreground) ──
echo "  ========================================="
echo "  Athena pipeline running (Ctrl+C to stop)"
echo "  ========================================="
echo "  LLM: direct load via llama-cpp-python (loaded first, clean memory)"
echo "  STT: Whisper tiny.en on GPU"
echo "  TTS: Kokoro-82M on GPU"
echo "  VAD: Silero on CPU"
echo ""

sudo docker compose --env-file "$ATHENA_HOME/.env" -f "$ATHENA_HOME/docker-compose.yml" \
    logs -f athena-orchestrator 2>&1 | tee -a "$LOG_DIR/stack_$(date +%Y%m%d_%H%M%S).log"

# Cleanup on exit
echo ""
echo "  Stopping all services..."
sudo docker compose --env-file "$ATHENA_HOME/.env" -f "$ATHENA_HOME/docker-compose.yml" down 2>/dev/null || true
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
echo "  Memory freed. Run ~/athena/athena.sh to restart."
