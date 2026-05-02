# Athena Xai v1 — Code Review Report

**Target hardware:** Jetson Orin Nano Super 8GB / JetPack 6.2 / L4T R36.4 / CUDA 12.6
**Pipeline:** Mic → Silero VAD (CPU) → Whisper STT (GPU) → Gemma 4 E2B abliterated LLM (GPU) → Kokoro-82M TTS (GPU) → USB Composite headset / wired UACDemoV1.0 speaker
**Treated as:** Fully functional v1 build. This is a "how it works" report — every subsystem is in place and verified working in production logs.

**Note on audio architecture:** No PulseAudio. Mic capture and TTS playback both use PortAudio (sounddevice) directly into ALSA → `/dev/snd`. No PA daemon, no socket, no cookie. PortAudio device indexes are looked up at startup by name pattern from `llm_config.json`.

**Note on the LLM:** llama-cpp-python loaded directly inside the orchestrator container (no separate LLM container, no HTTP hop). The model is `Huihui-gemma-4-E2B-it-abliterated-v2.Q4_K_M.gguf` (~1.7 GB Q4_K_M weights). Sampling uses Gemma 4's officially recommended params only (`temperature=1.0`, `top_p=0.95`, `top_k=64`); no penalties, no `max_tokens` cap.

**Note on translation:** Path A (foreign speaker → English) plays on the headset; Path B (English speaker → foreign) plays on the wired USB speaker. Foreign-script targets (Chinese, Japanese, Hindi, etc.) get a **two-pass LLM treatment** — translate to native script, then a second pass converts the foreign characters to English letters that the English espeak phonemizer can speak. Voice for those targets is `am_michael` (male) or `af_heart` (female), baked into `path_b_voice_table` directly. Latin-script targets (Spanish, French, Italian, Portuguese, etc.) take the single-pass path with their native lang code through espeak.

**Note on thinking:** No dedicated thinking mode. A per-turn trigger detects the whole word `"think"` anywhere in the user's utterance (conversation or adult mode only) and prefixes `"<|think|>"` onto that one LLM call's system prompt. Next turn reverts unless `"think"` is said again. Translate and idle never swap system prompts.

---

## Table of Contents

1. `athena.sh`
2. `docker-compose.yml`
3. `Dockerfile.orchestrator`
4. `Dockerfile.whisper`
5. `Dockerfile.tts`
6. `orchestrator/athena_pipeline.py`
7. `orchestrator/audio_gateway.sh`
8. `tts/tts_server.py`
9. `chat_athena.sh`
10. `test_athena.sh`
11. `stop_athena.sh`
12. `memory_diag.sh`
13. `preview_voices.sh`
14. `llm_config.json`
15. `constraints-docker.txt`
16. `requirements-orch.txt`
17. `requirements-tts.txt`
18. `Athena.desktop`
19. `README.md`
20. `FILES.md`
21. `athena-icon-*.png`
22. **Note A — Chat template and Gemma 4 channel tokens**
23. **Note B — Memory layout and headroom**
24. **Note C — The voice router state machine**

---

## 1. `athena.sh` — Master build + launch

### Name, location, purpose
- **Name:** `athena.sh`
- **Location:** Project root → `$ATHENA_HOME=$HOME/athena/athena.sh` after first run.
- **Purpose:** A 1036-line bash script. Cleans state, sizes swap, builds CUDA-enabled binaries on the host, downloads models, builds three Docker images, runs hardware tests, launches the stack in phased order, and tails the orchestrator logs in foreground.

### Connection / functions exposed
This is the entry point. Everything else is downstream. It exposes three reporting helpers, one image-build cache helper, and installs a Ctrl+C trap for graceful container shutdown.

- **`report_pass <step>`** — appends `[PASS]` to `$REPORT`. Saved to `logs/build_report.log`.
- **`report_fail <step>`** — appends `[FAIL]`, also echoes to stdout.
- **`report_skip <step>`** — appends `[SKIP]` for steps the user opted out of.
- **`build_if_changed <image_name> <dockerfile> [deps...]`** — md5-hashes the Dockerfile + listed deps, compares against `$ATHENA_HOME/.build-cache/<image>.md5`. Skips only if the hash matches AND `docker image inspect` succeeds. Otherwise runs `docker build --no-cache`.
- **`cleanup_and_exit`** — Ctrl+C trap. Compose down + cache drop + compaction. Prevents orphaned containers when the operator interrupts the foreground tail.

### The 13 phases, in order

**Phase 1 — System cleanup.** Compose down (if files exist), `docker stop` + `rm -f` on all expected container names, `fuser -k /dev/nvhost-{ctrl,gpu}` to evict GPU device fd holders, full Docker GC, `apt-get clean`, `journalctl --vacuum-size=100M`, `sync; echo 3 > /proc/sys/vm/drop_caches`, `echo 1 > /proc/sys/vm/compact_memory`. Defensive removal of any stray model files or images that aren't part of the current set.

**Phase 2 — Scripts + desktop icon.** `chmod +x` on the six host helpers (`stop_athena.sh`, `test_athena.sh`, `chat_athena.sh`, `preview_voices.sh`, `memory_diag.sh`, `orchestrator/audio_gateway.sh`). If `~/Desktop` exists, `sed`-substitutes the icon path in `Athena.desktop`, copies to the desktop, marks trusted via `gio set ... metadata::trusted true`. If running outside `$ATHENA_HOME`, `cp -r` the source tree there.

**Phase 3 — Directories + power mode.** `mkdir -p` for `build/`, `models/{llm,whisper,tts,vad}`, `logs/`, `docker-staging/{bin,lib,cuda-libs,orchestrator,tts}`. `nvpmodel -m 0` for MAXN_SUPER 25W. `jetson_clocks` to lock max clocks.

**Phase 4 — NVMe swap + memory tuning.** Disables zram (uses RAM, useless for relieving CUDA pressure). Manages a 16 GB `/swapfile.athena`: `fallocate -l 16G` with `dd` fallback, `mkswap`, `swapon`. VM tuning: `swappiness=100`, `vfs_cache_pressure=500`, `dirty_background_ratio=1`, `dirty_ratio=5`, `min_free_kbytes=262144` (256 MB CUDA reserve).

**Phase 5 — Docker nvidia runtime.** `docker info | grep nvidia`. If missing, writes `/etc/docker/daemon.json` with `default-runtime: nvidia` and restarts Docker. The only step that hard-exits on failure — without nvidia runtime, nothing further is meaningful.

**Phase 6 — Build dependencies.** `dpkg -s` checks for `cmake build-essential git curl wget`. Apt-installs the full dev set if anything is missing.

**Phase 7 — Build llama.cpp + llama-cpp-python wheel.** Two sub-phases:
- (a) Clone llama.cpp shallow, `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release`, build the `llama-server` target with `-j2` (conservative for 8 GB RAM), stage binary + `.so` files to `docker-staging/`.
- (b) Builds the `llama-cpp-python` wheel on the host where the full CUDA toolkit is available. `pip install --upgrade pip setuptools` then `pip install scikit-build-core>=0.9,<0.10 cmake ninja exceptiongroup<1.3` — the `exceptiongroup<1.3` pin avoids the Python 3.12 `TypeVar(default=...)` crash on JetPack's Python 3.10. Build env: `CMAKE_BUILD_PARALLEL_LEVEL=1` (single-thread compile to avoid OOM freeze), `CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=87"`, `FORCE_CMAKE=1`. `pip3 wheel llama-cpp-python -w docker-staging/`. Three-tier fallback if `pip wheel` fails: try `pip install` then re-run `pip wheel --no-deps`, then last-resort find in `~/.cache/pip`.

**Phase 8 — Build whisper.cpp with CUDA.** Same clone-and-cmake pattern. `-DWHISPER_BUILD_SERVER=ON`. Looks for the binary under both `whisper-server` and `server` (name varies by upstream commit). Stages binary + every `*.so*`. The on-device `quantize` invocation produces the small Q8_0 model from the downloaded f16 base.

**Phase 9 — Download all models.** Five downloads, each guarded by `if [ -f X ] && [ -s X ]; then echo cached; else download; fi`:
- **Gemma 4 E2B abliterated v2 Q4_K_M GGUF** (~1.7 GB, requires `HF_TOKEN` for huggingface.co/Huihui-AI repository).
- **Whisper small base** (~480 MB f16 from huggingface.co/ggerganov/whisper.cpp), then on-device quantize to Q8_0 via the Phase-8 binary.
- **Silero VAD ONNX** (~2 MB) from snakers4 GitHub.
- **Kokoro-82M ONNX** (~350 MB) and voice pack (~27 MB) from `thewh1teagle/kokoro-onnx` releases.
- **onnxruntime-gpu Jetson wheel** (~70 MB) from `pypi.jetson-ai-lab.dev` with a `github.com/ultralytics/assets` mirror fallback.

**Phase 10 — Stage CUDA toolkit libraries from host.** Searches `/usr/local/cuda/lib64`, `/usr/local/cuda-12.6/targets/aarch64-linux/lib`, `/usr/lib/aarch64-linux-gnu`. Lib pattern allow-list: `libcudart.so*`, `libcublas.so*`, `libcublasLt.so*`, `libcudnn*.so*`, `libcufft.so*`, `libcurand.so*`, `libcusparse.so*`, `libcusolver.so*`, `libnvrtc.so*`, `libnvJitLink.so*`, `libcupti.so*`, `libnvinfer.so*`, `libnvonnxparser.so*`. `cp -P` preserves symlinks. Required: `libcudart`, `libcublas`, `libcublasLt` must all be present.

**Phase 11 — Build 3 Docker images.** Copies all Dockerfiles, requirements, and source code into `docker-staging/`. `build_if_changed` three times: `athena-whisper`, `athena-tts` (deps include requirements-tts, constraints, tts_server.py), `athena-orch` (deps include requirements-orch, constraints, athena_pipeline.py). The build report is printed and saved.

**Phase 12 — Audio hardware tests.** USB card detection via `pactl list cards short | grep -i usb` (used only for the audio test phase, not at runtime). Speaker test: generates a 440Hz, 1.5s, stereo 16-bit WAV inline with Python `wave + struct + math`, plays via `paplay` (with `aplay -q` fallback). Y/N. Mic test: records 5s, runs inline Python computing RMS over int16 samples. Buckets: RMS > 200 PASS, 50–200 LOW, < 50 FAIL.

**Phase 13 — Launch stack.** Three container phases:

- Stage `docker-compose.yml`, `llm_config.json`, write `.env` with `ATHENA_HOME`, `NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility`. `docker compose down` for safety. **Ctrl+C trap installed** — `cleanup_and_exit` brings everything down on SIGINT.
- **Phase 1 — RAM evacuation:** kill GPU fds, drop caches, compact memory. Then a Python pressure block: `[ctypes.create_string_buffer(256 * 1024 * 1024) for _ in range(14)]` — 14 chunks × 256 MB = 3.5 GB of pressure held for 3 seconds, then freed. Compaction afterward. This forces non-CUDA pages out before the LLM allocation.
- **Phase 2 — Orchestrator first:** `docker compose up -d athena-orchestrator`. The LLM is the largest single allocation (~1.7 GB Q4_K_M weights + KV cache + scratch), gets first dibs on clean contiguous memory. Sleep 20 seconds. Memory snapshot logged via `memory_diag.sh` to `logs/memory_phases.log`.
- **Phase 3 — TTS + Whisper together:** `docker compose up -d athena-tts athena-whisper`. Both come up in parallel. Memory snapshot.
- **Foreground tail:** `docker compose logs -f athena-orchestrator | tee -a logs/stack_<timestamp>.log`. User stays attached. On exit (any path), the cleanup at the bottom does `docker compose ... down` and drops kernel caches.

---

## 2. `docker-compose.yml` — 3-service stack

### Name, location, purpose
- **Name:** `docker-compose.yml`
- **Location:** Project root → `$HOME/athena/docker-compose.yml` at launch time.
- **Purpose:** Declares three services — `athena-tts`, `athena-whisper`, `athena-orchestrator` — with their images, volumes, env vars, and inline `bash -c` startup scripts.

### Connection / functions
Driven by `athena.sh` Phase 13. No functions; declarative YAML with shell command blocks worth annotating.

### Service-by-service

**`athena-tts`**
- `image: athena-tts:latest`. `network_mode: host`, `runtime: nvidia`, `restart: on-failure:1`.
- Volumes: `${ATHENA_HOME}/models:/models:ro`, `${ATHENA_HOME}/logs:/logs`.
- Env: `LD_LIBRARY_PATH=/usr/local/lib/cuda`.
- Inline command: `sleep 5` (GPU settle), `ldconfig`, banner, `ls -lh /models/tts/`, prints ORT providers, `exec python3 /app/tts/tts_server.py --port 8002 --model-dir /models/tts | tee -a /logs/tts.log`.
- Healthcheck: `curl -f http://localhost:8002/health` every 10s, 60s grace.

**`athena-whisper`**
- Same network/runtime/restart pattern. `shm_size: 256m`.
- Inline command: same banner + diagnostics, then `exec whisper-server --model /models/whisper/ggml-small-q8_0.bin --port 8001 --host 0.0.0.0 --threads 4 --processors 1 --print-progress --print-realtime --convert | tee -a /logs/whisper.log`.
  - Uses the locally-quantized `ggml-small-q8_0.bin`.
  - **Multilingual model** — language detection happens per request via the `language` form field (`auto` for Path A foreign-detection, `en` for everything else).
  - `--threads 4 --processors 1` — 4 CPU threads inside 1 inference processor.
- Healthcheck: 30 retries × 10s = 300s total, 60s grace — generous because the small model takes longer to load than tiny.en.

**`athena-orchestrator`**
- `restart: "no"` — deliberately not on-failure. A pipeline crash leaks a partial CUDA context; restarting allocates a second one on top. Crash → stays down → user re-runs `athena.sh` for proper cleanup.
- `shm_size: 1g`. `ulimits: memlock: -1, stack: 67108864` (unlimited mlock for LLM weight pinning, 64 MB stack).
- `devices: /dev/snd:/dev/snd` (ALSA passthrough). **No PulseAudio socket, no cookie mount** — PortAudio talks directly to ALSA via the device-node passthrough.
- Volume mounts: models read-only, logs writable, **`llm_config.json` mounted read-only at `/app/llm_config.json`** (so editing host-side and re-launching reloads tunables without rebuilding the image).
- Env: pipeline tuning via `ATHENA_*` vars: LLM model path, Whisper URL, TTS URL, log dir, VAD model path, voice + speed defaults, VAD threshold 0.5, silence timeout 1.0s, mic rate 48000, Whisper rate 16000, min speech frames 8. Plus `GGML_CUDA_NO_PINNED=1`.
- Inline command: ldconfig, prints ORT providers, prints `llama-cpp-python` version, prints contents of `llm_config.json` line by line, then `python3 /app/orchestrator/athena_pipeline.py | tee -a /logs/orchestrator.log`.

---

## 3. `Dockerfile.orchestrator`

### Name, location, purpose
- **Name:** `Dockerfile.orchestrator`
- **Location:** Project root → copied into `docker-staging/` at build time.
- **Purpose:** Builds `athena-orch:latest`. Self-contained: own copy of CUDA toolkit libs, own onnxruntime-gpu, own llama-cpp-python (installed from a host-built wheel — no compile inside Docker).

### Connection / functions
Built by `athena.sh` Phase 11 via `build_if_changed "athena-orch" "Dockerfile.orchestrator" requirements-orch.txt constraints-docker.txt orchestrator/athena_pipeline.py`. No functions.

### Stage-by-stage

1. **`FROM ubuntu:22.04`** — bare base.
2. **`apt-get install`**: `python3 python3-pip python3-dev portaudio19-dev libsndfile1 alsa-utils curl ca-certificates`. PortAudio + ALSA, no PulseAudio.
3. **`COPY cuda-libs/ /usr/local/lib/cuda/`** + register in `ld.so.conf.d/cuda-toolkit.conf` + `ldconfig`.
4. **`COPY constraints-docker.txt + ENV PIP_CONSTRAINT`** — every subsequent `pip install` reads this constraint file (the anti-torch wall).
5. **`pip install numpy>=1.24,<2.0`** — Jetson onnxruntime-gpu 1.20.x is built against numpy 1.x ABI.
6. **`pip install --no-deps onnxruntime_gpu-*.whl`** — staged Jetson wheel. `--no-deps` blocks the generic onnxruntime from being pulled over the top.
7. **`pip install llama_cpp_python-*.whl`** — host-built CUDA-baked wheel.
8. **`pip install -r requirements-orch.txt`** — `scipy 1.15.3`, `sounddevice 0.5.5`, `soundfile 0.13.1`, `requests 2.33.0`, `cffi 2.0.0`.
9. **Build verification** — multi-line `python3 -c`. Imports numpy, onnxruntime (asserts `'CUDAExecutionProvider' in get_available_providers()`), uses `importlib.metadata.version('llama-cpp-python')` instead of `import llama_cpp` (importing would `dlopen` `libllama.so` which links against `libcuda.so.1`, only available at container runtime via `runtime: nvidia` — not during `docker build`). Imports scipy, sounddevice. Prints `BUILD VERIFICATION: ALL PASS`.
10. **`COPY orchestrator/ /app/orchestrator/`**, **`WORKDIR /app`**, **`ENV LD_LIBRARY_PATH=/usr/local/lib/cuda`**.

---

## 4. `Dockerfile.whisper`

### Name, location, purpose
- **Name:** `Dockerfile.whisper`
- **Location:** Project root → `docker-staging/`.
- **Purpose:** Builds `athena-whisper:latest`. Leanest of the three: no Python, no pip — just the C++ `whisper-server` binary and its `.so` deps.

### Connection / functions
Built in Phase 11. The binary + `.so` files were compiled in Phase 8 on the host and staged.

### Stage-by-stage

1. `FROM ubuntu:22.04`.
2. `apt-get install curl ca-certificates libgomp1 ffmpeg`.
   - `libgomp1` — GNU OpenMP runtime. whisper.cpp is compiled with `-fopenmp`.
   - `ffmpeg` — required by `whisper-server --convert` for input format normalization.
3. `COPY cuda-libs/` → `/usr/local/lib/cuda/`. `COPY lib/` (staged ggml `.so` family) → `/usr/local/lib/`.
4. `COPY bin/whisper-server /usr/local/bin/whisper-server` + `chmod +x`.
5. Two `.conf` files in `/etc/ld.so.conf.d/` + `ldconfig` to register both lib directories.
6. `RUN whisper-server --help | head -3` — fails the build if the binary doesn't link properly.
7. `ENV LD_LIBRARY_PATH=/usr/local/lib/cuda:/usr/local/lib`.

---

## 5. `Dockerfile.tts`

### Name, location, purpose
- **Name:** `Dockerfile.tts`
- **Location:** Project root → `docker-staging/`.
- **Purpose:** Builds `athena-tts:latest`. Self-contained Kokoro-82M ONNX TTS server. **No torch. No misaki.**

### Connection / functions
Built in Phase 11. The 5-step layered install order matters.

### Stage-by-stage

1. `FROM ubuntu:22.04`.
2. `apt-get install python3 python3-pip python3-dev libsndfile1 espeak-ng curl ca-certificates`.
   - **`espeak-ng` is the phonemizer backend Kokoro uses** — Kokoro is phoneme-based; without espeak-ng for grapheme-to-phoneme conversion, synthesis fails. The orchestrator's two-pass for non-Latin Path B targets exists specifically to feed espeak only English-letter text.
3. `COPY cuda-libs/` → register → ldconfig.
4. `COPY constraints-docker.txt` + `ENV PIP_CONSTRAINT`. Same anti-torch wall.
5. **Step 1 — `pip install kokoro-onnx==0.5.0`** — pulls onnxruntime CPU + numpy 2.x as transitive deps.
6. **Step 2 — `pip install --force-reinstall numpy>=1.24,<2.0`** — force back to 1.x ABI.
7. **Step 3 — Swap onnxruntime CPU → onnxruntime-gpu Jetson wheel.** `pip uninstall -y onnxruntime`, then `pip install --no-deps onnxruntime_gpu-*.whl`. The `--no-deps` is critical — without it, pip's resolver would re-pull generic onnxruntime to satisfy upstream constraints.
8. **Step 4 — `pip install -r requirements-tts.txt`** — `soundfile 0.13.1`, `requests 2.33.0`.
9. **Step 5 — Build verification** — imports numpy, onnxruntime (asserts CUDA EP available), `from kokoro_onnx import Kokoro`, imports soundfile. Prints `BUILD VERIFICATION: PASS`.
10. `COPY tts/ /app/tts/`, `WORKDIR /app`, `ENV LD_LIBRARY_PATH=/usr/local/lib/cuda`.

---

## 6. `orchestrator/athena_pipeline.py` — The runtime

### Name, location, purpose
- **Name:** `athena_pipeline.py`
- **Location:** `orchestrator/athena_pipeline.py` → baked into `athena-orch:latest` at `/app/orchestrator/athena_pipeline.py`.
- **Purpose:** 1983-line Python runtime. Owns the audio loop, VAD, LLM (loaded directly via `llama-cpp-python`), HTTP calls to TTS and Whisper, conversation memory, the voice router state machine, the persistence layer for Settings, the dual-channel translate router, the per-turn thinking trigger, and the Asian double-pass for non-Latin Path B targets.

### Connection / functions
Not invoked by `athena.sh`. Started by the orchestrator container's compose `command:`. Reads `/app/llm_config.json` and the `ATHENA_*` env vars at startup.

### Module-level constants (key values)

- **`MAX_CONVERSATION_MESSAGES = 20`** — sliding window for conversation history. The LLM call always sees system + last 20 messages, with older ones trimmed.
- **`PRE_BUFFER_CHUNKS = 16`** (~512 ms) — pre-speech ring buffer of 32-ms mic chunks. When VAD confirms speech, this ring is prepended to the recording so we don't clip word-onsets.
- **`MIC_RATE = 48000`**, **`VAD_RATE = 16000`**, **`WHISPER_RATE = 16000`**, **`VAD_CHUNK = 512`** samples (32 ms at 16 kHz), **`VAD_CONTEXT = 64`** samples (Silero v5 requires the previous chunk's last 64 samples prepended).
- **`MIN_SPEECH_FRAMES = 8`** (~256 ms) and **`MIN_SPEECH_DURATION = 0.3` s** — confirmation thresholds.
- **`SILENCE_TIMEOUT = 1.0` s** — speech-end declaration.
- **`VAD_THRESHOLD = 0.5`** — Silero confidence cutoff.
- **`TTS_MUTE_DURATION = 5.0` s** — post-stop mute window.
- **`WAKE_PHRASES = ["hey athena", "okay athena"]`**, **`STOP_PHRASES = ["athena stop", "thank you athena"]`**.
- **`TRANSLATE_WAKE = "athena translation mode"`**, **`TRANSLATE_STOPS = ["athena stop translating", "athena stop translation", "athena stop translate"]`**.
- **`ADULT_WAKE = "athena adult mode"`**, **`ADULT_STOP = "athena normal mode"`**.
- **`SETTINGS_WAKE = "athena settings"`**, **`SETTINGS_EXIT_PHRASES = ["exit settings", "settings exit", "done", "athena exit"]`**.
- **`TRANSLATE_UNLOCK_PHRASES = ["athena unlock language", "athena clear language"]`**.
- **`ATHENA_WAKE_VARIANTS = ["athena", "a theme a", "a theme", "the theme"]`** — fuzzy matching for Whisper's chronic "athena" → "a theme" mishears.
- **`ASIAN_OVERRIDE_PROMPT = "Only phonemize this to English characters. do not respond and do not translate to English."`** — verbatim user-supplied prompt for the second LLM pass on non-Latin Path B targets.
- **`LANGUAGE_MAP`** — 100-entry dict mapping spoken language names ("spanish", "japanese", etc.) to ISO codes for Whisper's translate-to-English Path A.

### `load_llm_config()` (line 78)
Returns a dict. Tries `open(LLM_CONFIG_PATH, "r")`, defaults to `/app/llm_config.json`. Merges file contents over a built-in defaults dict (`{**defaults, **cfg}`) — file values win on collision. Returns defaults on `FileNotFoundError`. The defaults are: system_prompt, temperature 1.0, top_p 0.95, top_k 64, n_ctx 4096, n_gpu_layers 99, n_batch 128, plus the audio device patterns, speed grid, voice options, and path_b_voice_table.

### `athena_match(text_lower, base_phrase)` and `athena_any_match(...)` (line 184)
Wake-word fuzzy matcher. If `base_phrase` contains `"athena"`, builds rewrites for each variant in `ATHENA_WAKE_VARIANTS` and returns True if any of them appears in `text_lower`. If the base phrase doesn't contain "athena" (e.g. plain "stop translating"), falls through to a literal substring match. The fuzz applies uniformly — every router branch that checks for "athena…" automatically accepts the misheard variants.

### `SileroVAD` class (line 252)
- **`__init__`** — Loads the Silero VAD ONNX from `/models/vad/silero_vad.onnx`. **Forces CPU** (`providers=["CPUExecutionProvider"]`, `intra_op_num_threads=1`). The reason: a CUDA session sometimes loaded but crashed on first inference with `CUBLAS_STATUS_ALLOC_FAILED` because the GPU was already mostly full from the LLM. ORT's auto-fallback only fires during session creation, not inference, so the only safe answer is "always CPU". VAD is <1 ms per chunk on CPU. Auto-detects model version by inspecting `session.get_inputs()` names: `"state"` key → v5/v6 (single combined LSTM state of shape `(2, 1, 128)`); `"h"` and `"c"` → v4. Allocates the appropriate state tensor(s) as zeros. Allocates a 64-sample context buffer. Runs a warm-up inference with all-zero input to JIT/compile the graph.
- **`_run_inference(chunk)`** — Branches on Silero model version. v5: `{"input": chunk, "sr": self._sr, "state": self._state}`, captures `out[1]` as new state. v4: `{"input": chunk, "sr": ..., "h": ..., "c": ...}`, captures `out[1]` and `out[2]`. Returns confidence scalar.
- **`process_chunk(audio_16k)`** — Validates `len == VAD_CHUNK (=512)`. Reshapes to `(1, 512)`. Concatenates `self._context (1, 64)` with the new chunk → `(1, 576)`. Calls `_run_inference`. Slides context window: `self._context = chunk[:, -64:]`. The Silero v5 `sr` field is `np.array(VAD_RATE, dtype=np.int64)` — a 0-d ndarray, not a scalar.
- **`reset()`** — Zeros LSTM state and context buffer.

### Number-to-words helpers (lines 322–379)
- **`_number_group_to_words(group_str)`** — converts a 1-3 digit group to English words.
- **`_expand_comma_number(match)`** — expands "5,280" → "five thousand two hundred eighty" up to trillions.
- **`_expand_decimal(match)`** — expands "3.14" → "three point one four".
- **`numbers_to_words(text)`** — applies both regexes. Called by `clean_for_tts`. Without this, Kokoro literally says "five comma two eight zero" and "one point three four" — broken pacing for spoken numbers.

### `split_for_tts(text, max_chars)` (line 381)
Splits long replies on sentence boundaries before sending to TTS so each chunk stays under Kokoro's preferred input size. Sentence boundary regex: `[.!?]\s+`. Falls back to character-count splits if no boundaries within `max_chars`.

### `clean_for_tts(raw)` (line 409)
Strips markdown, special tokens, expands numbers; does NOT strip thinking content. Pipeline:
- bold `**x**` → `x`, italic `*x*` → `x`, underline `__x__` → `x`
- markdown headers `^#{1,6}\s*` → empty
- fenced code blocks → empty
- inline code → unwrap
- list bullets `^[\s]*[-*+]\s+` → empty
- numbered lists `^[\s]*\d+\.\s+` → empty
- parenthetical content `\(.*?\)` → empty (drops parenthetical stage directions that sound bad spoken)
- bracket content `\[.*?\]` → empty
- residual special tokens like `<|channel>`, `<|/channel|>`, `<|im_end|>` regex'd out
- punctuation normalization: `=` → " equals ", `+` → " plus ", orphan `*`/`/`/`\\`/`_` → space (reduces Kokoro word-count mismatch warnings)
- numbers expanded via `numbers_to_words`
- newlines → spaces, runs of whitespace → single space

### Audio helpers (lines 441–456)
- **`resample_audio(audio, from_rate, to_rate)`** — identity if equal; else `scipy.signal.resample`. Returns float32.
- **`audio_to_wav_bytes(audio_f32, sr)`** — In-memory `BytesIO` + `wave.open(buf, "wb")` mono 16-bit WAV. Clips at ±32767. Returns raw bytes.

### `send_to_whisper(wav_bytes, language_override=None, translate_override=None, duration_hint=None)` (line 457)
POSTs to `WHISPER_URL+"/inference"`:
- `data={"response_format": "verbose_json", "temperature": "0.0", "language": <"auto"|"en"|...>}`. An empty-string language is rewritten to `"auto"` before sending so whisper-server actually runs auto-detection rather than defaulting to a specific language.
- `data["translate"] = "true"` if `translate_override` is True (Path A foreign→English translation server-side).
- `timeout = max(30, int(duration_hint * 2) + 15)` so a 2-minute clip gets a ~4-minute timeout. No hard cap on input audio length; whisper.cpp internally chains 30-second windows.
- Returns `(text, detected_language)`.

### `load_llm()` (line 491)
Constructs `Llama()` with the model path, `n_ctx`, `n_gpu_layers=99`, `n_batch=128`, `verbose=True`. Logs load time and model metadata. Returns the LLM handle.

### `_parse_speed_step(text_lower)` (line 517)
Parses spoken numbers ("one" through "nine", plus "1" through "9", plus "step five" etc.) into integer 1-9 for the Settings menu speed selectors. Returns `None` if not parsable.

### `send_to_llm(text, conversation, llm)` (line 530)
Appends `{"role": "user", "content": text}` to conversation. Calls `llm.create_chat_completion(messages=conversation, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k)` — Gemma 4 official sampling only. Pulls reply, finish_reason, prompt_tokens, completion_tokens. Computes `tok_s = completion / elapsed`. Logs the per-turn block:
```
PIPELINE <<< LLM ({elapsed}s)
  Prompt:   {n} tokens
  Reply:    {n} tokens @ {tok_s} tok/s
  Reply:    {words} words
  Finish:   {reason}
  Text:     '{first 80 chars}'
PIPELINE [llm-raw-full] bytes={n} :: {repr(reply)}
```
The `[llm-raw-full]` line uses Python's `repr()` so special tokens, escape sequences, and Unicode escapes are visible in the log. Saves `reply` verbatim to conversation (no extraction, no branching). Sliding window: appends assistant reply, then `if len(conversation) > MAX_CONVERSATION_MESSAGES + 1: del conversation[1:len(conversation) - MAX_CONVERSATION_MESSAGES]`. Returns the raw reply string.

### `send_to_tts(text, voice=None, speed=None, lang=None)` (line 589)
JSON POST to `TTS_URL+"/synthesize"` with `{"text", "voice", "speed", "lang"}`. Per-call overrides: voice/speed default to active session settings; lang defaults to `"en-us"`. Timeout 30 s. Returns raw WAV bytes or `None`.

### `kill_active_paplay()` (line 613)
Thin wrapper around `sounddevice.stop()`. Vestige from the PulseAudio era — `sd.play()` is per-chunk and self-contained, but the name and the call site survive in case anything ever needs to abort an in-flight play.

### `load_runtime_state()` and `save_runtime_state(state)` (lines 621, 635)
Read/write the persisted Settings file at `/logs/.athena_runtime.json` (host-side `~/athena/logs/.athena_runtime.json`). Keys: `_version`, `active_output_sink_label`, `voice_name`, `voice_speed_step`, `translation_speed_step`, `translation_voice_gender`. Missing keys are filled in via `setdefault` so a partially-written file still loads cleanly.

### `find_pa_device(pattern, kind="input")` (line 644)
Walks `sd.query_devices()` looking for the first device whose name contains `pattern` (case-insensitive substring match) and has the requested channel direction (input ≥ 1 or output ≥ 1). Returns the PortAudio index or `None`. Used at startup to lock the headset's mic + speaker and the wired USB speaker by their config-driven name patterns (`"Composite Device"`, `"UACDemoV1"`).

### `resolve_path_b_voice(target_language, gender)` (line 664)
Looks up the chosen translate target in `path_b_voice_table` (or `_fallback`) and returns `(lang_code, voice_id, bucket)` based on the user's translation gender setting. The non-Latin entries in the table carry `lang="en-us"` and use American voices (`am_michael` / `af_heart`) directly, so Path B for chinese/japanese/hindi/etc. resolves to English-voice + en-us at this step before any double-pass logic fires.

### `play_to_device(wav_bytes, device_idx)` (line 686)
Decodes the WAV, resamples 24 kHz → device rate via `scipy.signal.resample`, calls `sd.play(audio, samplerate=device_rate, device=device_idx); sd.wait()`. Logs `[audio-monitor] sd.play device={N} src={Hz} → {Hz} ch=1 duration={s}`.

### `stream_llm_to_tts(text, conversation, llm, device_idx, voice, speed, lang, on_first_chunk)` (line 742)
The streaming dispatcher used by conversation/adult mode. Calls `llm.create_chat_completion(stream=True)`. As tokens arrive:
- Accumulate into `full_raw` and `reply_buffer`.
- On every `\n`, split off completed sentences and push them to a `sentence_q` (a Python `queue.Queue`).
- A worker thread pulls from `sentence_q`, runs each sentence through `send_to_tts` + `play_to_device`, and signals `on_first_chunk` after the first sentence has been queued (for VAD unmute on first audio).
- At end-of-stream, flushes any remaining buffer to the queue. No streaming gate, no extraction — every chunk flows to TTS.
- After the loop, saves `full_raw` verbatim to conversation history and logs the per-turn stats.

### `memory_snapshot(label, throttle_seconds=0)` (line 917)
Writes an in-container memory snapshot to `/logs/memory_phases.log`. Reads `/proc/meminfo` and `/proc/buddyinfo` directly — container view. Throttled at 15 s per call so a fast back-and-forth session doesn't fill the log. Called at warm-idle (right after the boot greeting plays) and after every VAD utterance end.

### `list_audio_devices()` (line 977)
Iterates `sd.query_devices()`, marks `[IN]` and `[OUT]` defaults, logs each device with its name, channel counts, and PortAudio index. Diagnostic — runs once at startup so the `pipeline.log` has a permanent record of which PortAudio index each USB device got assigned.

### `wait_for_services()` (line 993)
Polls `TTS_URL+"/health"` for up to 90 retries × 2 s (3 minutes), then `WHISPER_URL+"/health"` for up to 60 retries × 2 s (2 minutes). Returns `(tts_ok, whisper_ok)`. Pipeline continues even if one or both fail (the orchestrator can still serve text-only conversations through `chat_athena.sh` for diagnosis).

### `main()` (line 1037)
The big control loop. Phases:

1. **Banner** — logs every config value at startup including the active LLM sampling, audio device patterns, voice router phrases, and persisted Settings.
2. **VAD load** — `vad = SileroVAD(...)`.
3. **LLM load** — `llm = load_llm()`. This is the single largest GPU allocation; the orchestrator container starts first specifically so this lands in clean contiguous memory.
4. **System prompt build** — `normal_system` and `adult_system` strings constructed from `llm_config.json`. Conversation initialized as `[{"role": "system", "content": normal_system}]`.
5. **PortAudio device lookup** — finds headset (input + output) and wired USB speaker (output) by name pattern from config. Logs the resolved PortAudio indexes.
6. **Persisted Settings restore** — voice, voice speed, translation speed, translation voice gender, active output sink (headset vs speaker).
7. **Wait for services** — TTS + Whisper health-check.
8. **Startup greeting** — `send_to_llm("Hello Athena", conversation, llm)` exercises the full chat-template path once and primes Gemma's KV cache. Reply → `clean_for_tts` → `send_to_tts` → `play_to_device(active_output)`.
9. **Warm-idle memory snapshot** — captures the steady-state baseline after the greeting plays.
10. **State machine init** — `is_speaking`, `vad_muted`, `silence_start`, `speech_buffer`, `pre_buffer` (deque maxlen 16), `consecutive_speech`, `speech_confirmed`, `is_idle`, `current_mode = "conversation"`, `last_active_mode = "conversation"`, `tts_muted`, `tts_muted_until`, settings substate vars, translate language locks.
11. **Mic open** — `sd.InputStream(samplerate=48000, channels=1, dtype="float32", blocksize=1536, device=headset_in)`. Each read is 1536 samples = 32 ms. Per chunk:
    - If `vad_muted` (TTS busy speaking), drop the chunk.
    - Resample 48k → 16k (1536 → 512). Truncate or zero-pad to exactly 512.
    - `confidence = vad.process_chunk(vad_input)`.
    - Maintain `pre_buffer` while idle; copy to `speech_buffer` on speech-confirmed transition.
    - When silence ≥ `SILENCE_TIMEOUT (1.0s)` and `len(speech) ≥ MIN_SPEECH_DURATION (0.3s)`: concatenate, resample to 16 kHz, encode WAV, call Whisper.
12. **Voice router** (next several hundred lines). The router's structure is documented in **Note C** below. Briefly: mode-aware dispatch on the transcript text, with one branch per phrase pattern. Translate mode has its own sub-state (waiting for language, language locked, Path A vs Path B per turn). Settings mode has its own sub-state (awaiting command, awaiting value for whichever option was selected). Mode entries/exits rebuild `conversation[0]["content"]` accordingly.
13. **LLM dispatch** — depending on mode and translate path:
    - **Conversation / Adult (streaming):** per-turn thinking trigger sets `conversation[0]["content"]` to either `normal_system`/`adult_system` or `"<|think|>" + base_prompt`. Then `stream_llm_to_tts(...)`.
    - **Conversation / Adult (non-streaming fallback):** same per-turn trigger logic, then `send_to_llm` + `clean_for_tts` + `send_to_tts` + `play_to_device`.
    - **Translate Path A** (foreign speaker → English on headset): Whisper called with `translate=True, language=auto`. Returns the English translation directly. Plays on headset device.
    - **Translate Path B** (English speaker → foreign on speaker): `reply = send_to_llm(forwarded_text, translate_conversation, llm)` translates to native script. If `tgt_bucket in ("chinese","japanese","hindi") and reply`, runs the second pass: builds a fresh one-shot conversation `[{"role":"system","content":ASIAN_OVERRIDE_PROMPT}, {"role":"user","content":reply}]`, calls `send_to_llm` again, replaces `reply` with the pass-2 output. Then `clean_for_tts` + `send_to_tts(voice=tgt_voice, lang=tgt_lang_code)` + `play_to_device(path_b_device)`. The voices for non-Latin buckets come from `path_b_voice_table` directly (`am_michael` / `af_heart`); no runtime override.
14. **Per-turn memory snapshot** — fires at every utterance end (throttled 15 s).
15. **`KeyboardInterrupt`** logs shutdown gracefully. Other exceptions log `FATAL` with `exc_info=True` and re-raise (container exits, `restart: "no"` keeps it down).

---

## 7. `orchestrator/audio_gateway.sh` — USB hot-plug monitor (unused)

### Name, location, purpose
- **Name:** `audio_gateway.sh`
- **Location:** `orchestrator/audio_gateway.sh` → baked at `/app/orchestrator/audio_gateway.sh`.
- **Purpose:** A bash daemon that watches PulseAudio for hot-plug events and re-applies stereo profile + default sink/source when the USB headset disconnects/reconnects.

### Connection / functions
Not invoked at runtime. Athena's audio path is PortAudio direct-to-ALSA, and `sd.query_devices()` enumerates the current device list on demand whenever the orchestrator needs it — hot-plug handling is built into the audio stack, no daemon needed. The script ships with the package for operators who run a PulseAudio-based audio stack and want the device-watching behavior in that environment.

---

## 8. `tts/tts_server.py` — Kokoro-82M HTTP server

### Name, location, purpose
- **Name:** `tts_server.py`
- **Location:** `tts/tts_server.py` → baked at `/app/tts/tts_server.py`.
- **Purpose:** 230-line Python HTTP server on port 8002. Wraps Kokoro-82M ONNX with `onnxruntime` CUDAExecutionProvider. **Hard-fails if CUDA isn't actually active** — no CPU fallback by design.

### Connection / functions
Started by the `athena-tts` compose command. Hit by orchestrator's `send_to_tts()` for every assistant turn, by `preview_voices.sh`, and by anything else POSTing `/synthesize`.

### Function-by-function

- **`init_kokoro(model_dir)`** — Imports onnxruntime + Kokoro lazily. Logs available providers.
  - **Hard-fails** if `'CUDAExecutionProvider' not in available` — raises `RuntimeError("CUDAExecutionProvider not available — TTS requires GPU, no CPU fallback")`.
  - Providers list with one entry: `("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE", "do_copy_in_default_stream": "1"})`. EXHAUSTIVE benchmarks every cuDNN convolution algorithm at startup and picks the fastest — slower startup, no "OP Conv running in Fallback mode" warnings during synthesis.
  - Creates the `InferenceSession`. Verifies CUDA is actually active via `sess.get_providers()`.
  - `kokoro_instance = Kokoro.from_session(sess, voices_path)`. Using `from_session` (not `Kokoro(model_path, voices_path)`) lets the server control the session's providers config exactly.
  - Lists voices, populates `engine_info`.

- **`ThreadingHTTPServer`** — single-line subclass mixing `socketserver.ThreadingMixIn` into `HTTPServer`. With `daemon_threads = True`, every request runs in its own thread. Healthchecks don't queue behind synthesis.

- **`TTSHandler` class**:
  - **`do_GET(self)`** — routes `/health` (returns `{status: ok, ...engine_info}`) and `/voices` (returns `{"voices": [...]}`). Anything else → 404.
  - **`do_POST(self)`** — only `/synthesize`. Reads `Content-Length` bytes, `json.loads`. Pulls `text`, `voice` (default `af_sky`), `speed` (default `1.0`), `lang` (default `"en-us"`). 400 if `text` empty.
    - Calls `kokoro_instance.create(text, voice=voice, speed=speed, lang=lang)` → `(samples, sample_rate)`. Single code path — no per-language routing in the server. The orchestrator delivers English-letter text on Asian/non-Latin paths via the two-pass flow.
    - Logs RTF (real-time factor = elapsed / audio_duration).
    - Encodes to WAV in-memory via `soundfile.write(buf, samples, sample_rate, format="WAV")`.
    - Returns `audio/wav` with proper Content-Length.
  - **`log_message(self, format, *args)`** — overrides stdlib's HTTP logger to route through the Python `log` logger.

- **`main()`** — argparse for `--port`, `--model-dir`, `--log-file`. Optional file handler. Banner. `init_kokoro(args.model_dir)`. `ThreadingHTTPServer(("0.0.0.0", args.port), TTSHandler).serve_forever()`.

---

## 9. `chat_athena.sh` — Text-only chat (bypass mic/TTS)

### Name, location, purpose
- **Name:** `chat_athena.sh`
- **Location:** Project root → `$HOME/athena/chat_athena.sh`.
- **Purpose:** REPL. Type prompts, get text replies. Bypasses VAD/Whisper/TTS — pure text in, text out. Useful for testing the LLM brain without microphone or speakers.

### Connection / functions
Just `chmod +x`'d. Manual run. Requires the orchestrator container to already be running.

### Function-by-function

No bash functions. Linear:
1. `docker ps | grep -q athena-orchestrator` — exits 1 if missing.
2. `while true; do read -p "You: " INPUT; ...`. Quits on `quit` or `exit`.
3. For each input, `docker exec athena-orchestrator python3 -c "<inline script>"`. The inline script:
   - `os.environ.setdefault('GGML_CUDA_NO_PINNED', '1')`.
   - Loads `/app/llm_config.json` into `cfg`.
   - **`Llama(model_path=..., n_ctx=cfg.get('n_ctx', 4096), ...)`** — fresh load on every prompt.
   - Builds `[system, user]` only — no chat history is preserved across REPL prompts.
   - `create_chat_completion`. Strips channel tokens. Prints reply.
4. Bash echoes `Athena: $RESPONSE`.

---

## 10. `test_athena.sh` — Service health probe

### Name, location, purpose
- **Name:** `test_athena.sh`
- **Location:** Project root → `$HOME/athena/test_athena.sh`.
- **Purpose:** Manual service health check. Run in a separate terminal while the stack is up.

### Connection / functions
Not in the pipeline. `athena.sh` only `chmod +x`'s it.

### Function-by-function

No functions. Three inline checks:
1. `curl -sf http://localhost:8001/health` — Whisper.
2. `curl -sf http://localhost:8002/health` — TTS.
3. `docker ps --format '{{.Names}}' | grep -q athena-orchestrator` — orchestrator container alive.

Each increments `$PASS` or `$FAIL`. Final summary line.

---

## 11. `stop_athena.sh` — Force-stop and free memory

### Name, location, purpose
- **Name:** `stop_athena.sh`
- **Location:** Project root → `$HOME/athena/stop_athena.sh`.
- **Purpose:** Heavy-handed shutdown. Used either standalone or by `athena.sh`'s Ctrl+C trap.

### Connection / functions
Manual. Also referenced by the cleanup paths in `athena.sh`.

### Function-by-function

No functions. Linear:
1. `SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"`, `ATHENA_HOME="$HOME/athena"`.
2. `docker compose down` (with file-existence guard).
3. `docker stop athena-whisper athena-tts athena-orchestrator` + `docker rm -f` on the same names.
4. `fuser -k /dev/nvhost-{ctrl,gpu}`.
5. `sync; echo 3 > /proc/sys/vm/drop_caches`, `echo 1 > /proc/sys/vm/compact_memory`.

---

## 12. `memory_diag.sh` — Memory + GPU + fragmentation diagnostic

### Name, location, purpose
- **Name:** `memory_diag.sh`
- **Location:** Project root → `$HOME/athena/memory_diag.sh`.
- **Purpose:** Multi-section memory health report. Used both **between launch phases** (the only script `athena.sh` invokes besides standard build tools) and **standalone** for debugging cudaMalloc behavior.

### Connection / functions
Phase 13 of `athena.sh` runs it three times into `logs/memory_phases.log`: before containers start, after the orchestrator+LLM are up (Phase 2 snapshot), after TTS+Whisper are up (Phase 3 snapshot). The orchestrator's in-process `memory_snapshot()` writes additional in-container snapshots at warm-idle and after every utterance end.

### Function-by-function

No functions. Sections (the `--brief` flag is parsed but unused):

0. **debugfs auto-mount** at the very top — `sudo mount -t debugfs debugfs /sys/kernel/debug` so `extfrag_index` and nvmap allocation files become readable. Idempotent and harmless if already mounted or unprivileged.
1. **RAM** — `free -m` parsed with awk into `Total/Used/Free/Available` and `Swap`. Note line: `(HEADROOM = 'Available', not 'Free' — Available counts reclaimable cache)`.
2. **/proc/meminfo (key fields)** — MemAvailable, Cached, Buffers, Mapped, Slab, SReclaimable, AnonPages, Shmem, Dirty, Writeback, KernelStack, PageTables, VmallocUsed, HugePages_Total, HardwareCorrupted.
3. **Swap Devices** — `swapon --show`, plus a warning if zram is active (zram uses RAM, doesn't help CUDA).
4. **VM Settings** — prints `swappiness`, `vfs_cache_pressure`, `dirty_ratio`, `dirty_background_ratio`, `min_free_kbytes`.
5. **Buddy Allocator** — `cat /proc/buddyinfo`, then per-zone reports the largest available contiguous block (orders 10 down to 6, where order 10 = 4 MB). The answer to "can the kernel give CUDA an N MB contiguous allocation?". Plus a per-zone order-10 (4 MB block) total summary.
6. **Fragmentation** — `/sys/kernel/debug/extfrag/extfrag_index` if available (0 = no frag, 1 = totally fragmented).
7. **GPU Memory** — `tegrastats --interval 1000 | head -1` snapshot, with the lfb (largest free block) field parsed out into its own line: `GPU lfb: {count} x {size}MB = {total}MB largest contiguous (CUDA-allocatable)`. `nvidia-smi --query-gpu=memory.total,used,free`. NvMap allocations from `/sys/kernel/debug/nvmap/iovmm/allocations` if mounted.
8. **Docker Container Memory** — `docker stats --no-stream | grep athena`, prefixed with a NOTE explaining that on Jetson unified memory, container MemUsage includes mmap'd model file pages which also count toward system Cached and MemAvailable — don't add container totals to process RSS to estimate pressure.
9. **Top 10 Memory Consumers** — `ps aux --sort=-%mem | head -11`.
10. **Headroom summary** — single-line summary using `MemAvailable` and parsed `lfb` total.

---

## 13. `preview_voices.sh` — Audition every Kokoro voice

### Name, location, purpose
- **Name:** `preview_voices.sh`
- **Location:** Project root → `$HOME/athena/preview_voices.sh`.
- **Purpose:** Audition tool. Plays a configurable test phrase through every Kokoro voice. For picking the value of `ATHENA_TTS_VOICE` env var or the Settings menu's voice option. Saves WAVs to `~/athena/voice_samples/`.

### Connection / functions
`chmod +x`'d only. Manual run, requires `athena-tts:8002` up.

### Function-by-function

- **`play_voice(voice, speed)`** — POSTs to `/synthesize` with the test phrase, voice, speed; saves WAV to `voice_samples/<voice>_speed<speed>.wav`. Logs duration via `soundfile.info`. Plays via `aplay -q`, with `paplay` and `python3 + soundfile/sounddevice` as cascading fallbacks.

- **Bash main:** health-checks the TTS server, fetches `/voices` JSON, parses voice names with inline `python3 -c` (no jq dependency). Three modes: Quick (every voice at 1.0×), Full (every voice at 0.8/1.0/1.2×), Pick (interactive numbered menu).

---

## 14. `llm_config.json`

### Name, location, purpose
- **Name:** `llm_config.json`
- **Location:** Project root → `$HOME/athena/llm_config.json` → mounted read-only at `/app/llm_config.json` inside the orchestrator.
- **Purpose:** Single source of truth for runtime tunables. Editing the host file and re-running `athena.sh` reloads tunables without rebuilding any Docker image (volume mount, not COPY'd).

### Connection / functions
Phase 13 stages it. Compose mounts it. `athena_pipeline.py:load_llm_config()` reads it at startup. Compose's startup banner echoes its contents. `chat_athena.sh` re-reads it for each one-shot call.

### Field-by-field

- **`system_prompt`** — conversation-mode system prompt. Personality only (formatting handled by `clean_for_tts`).
- **`adult_prompt`** — adult-mode system prompt. Flirty/companion personality.
- **`translate_prompt`** — pass-1 translate system prompt template, with `{language}` formatted at runtime to the locked target language.
- **`enable_thinking`** (false) — would prefix `"<|think|>"` to the conversation system prompt. Not used as the default; the per-turn trigger handles thinking.
- **`adult_thinking`** (false) — same for adult mode.
- **`think_trigger_phrase`** ("think") — the whole-word match for the per-turn thinking trigger.
- **`streaming`** (true) — use `stream_llm_to_tts` for conversation/adult turns.
- **`headset_alsa_pattern`** ("Composite Device") — substring used by `find_pa_device` to lock the wireless headset.
- **`external_alsa_pattern`** ("UACDemoV1") — substring for the wired USB speaker.
- **`path_b_force_external_device`** (true) — Path B always plays on the wired speaker, even if `external_device` lookup at startup returned the headset by accident.
- **`translate_lock_consecutive`** (2) — translate mode auto-locks the language after 2 consecutive same-language detections.
- **`tts_max_chars_per_chunk`** (300) — input split-size for `split_for_tts`.
- **`voice_options`** — `{"sky": "af_sky", "heart": "af_heart", ...}`. Friendly nicknames mapping to Kokoro voice IDs for the Settings menu.
- **`speed_grid`** — `[0.68, 0.76, 0.84, 0.92, 1.00, 1.08, 1.16, 1.24, 1.32]`. The 1-9 step values. Step 5 = 1.00.
- **`path_b_voice_table`** — the big map. ~30 entries grouped by phonetic bucket:
  - Latin script (single-pass): spanish, italian, portuguese, french, plus phonetic-fallback group-mates (tagalog/indonesian/malay/catalan in spanish bucket; romanian/latin/maltese in italian bucket; etc.). Each entry carries a native lang code (`es`, `it`, `pt-br`, `fr-fr`) and a native voice (`ef_dora`/`em_alex`, `if_sara`/`im_nicola`, `pf_dora`/`pm_alex`, `ff_siwis`/`em_alex`).
  - Non-Latin (double-pass): chinese, japanese, korean (japanese bucket), and the entire hindi bucket (hindi, urdu, bengali, punjabi, marathi, gujarati, nepali, sanskrit, sinhala, sindhi, assamese, persian, pashto, tajik, tamil, telugu, kannada, malayalam), plus the chinese-bucket non-zh entries (vietnamese, thai, lao, burmese, khmer, tibetan). Every non-Latin entry carries `lang="en-us"` and uses `am_michael` (male) / `af_heart` (female) directly.
  - `_fallback` — `lang="en-us"`, `am_michael`/`af_sky`. Used when the spoken target language doesn't match any table key.
- **`translation_voice_gender_default`** ("male") — initial gender selection until the user changes it via Settings.
- **`temperature`** (1.0), **`top_p`** (0.95), **`top_k`** (64) — Gemma 4 official sampling.
- **`n_ctx`** (4096) — KV cache size at load.
- **`n_gpu_layers`** (99) — all layers to GPU.
- **`n_batch`** (128) — prompt-processing batch size.

---

## 15. `constraints-docker.txt`

### Name, location, purpose
- **Name:** `constraints-docker.txt`
- **Location:** Project root → copied into `docker-staging/` and ingested by all three Dockerfiles.
- **Purpose:** Pip constraint file. Blocks `torch`, `torchaudio`, `torchvision` from ever being installed even as transitive deps.

### Connection / functions
Each Dockerfile sets `ENV PIP_CONSTRAINT=/tmp/constraints.txt` so every `pip install` in the build picks it up.

### Per-line

```
torch==99999.0.0
torchaudio==99999.0.0
torchvision==99999.0.0
```

Pinning to a non-existent version forces pip's resolver to refuse rather than silently downgrade. Anything that needs torch fails install instead of pulling a wrong CUDA-13-flavored wheel.

---

## 16. `requirements-orch.txt`

### Per-line

`scipy==1.15.3`, `sounddevice==0.5.5`, `soundfile==0.13.1`, `requests==2.33.0`, `cffi==2.0.0`. Numpy and onnxruntime-gpu are installed separately in `Dockerfile.orchestrator` so versions are pinned exactly.

---

## 17. `requirements-tts.txt`

### Per-line

`soundfile==0.13.1`, `requests==2.33.0`. Kokoro and onnxruntime-gpu are installed separately in `Dockerfile.tts`. Numpy is force-installed at <2.0 in the same Dockerfile.

---

## 18. `Athena.desktop`

### Per-line

```
[Desktop Entry]
Name=Athena
Comment=Voice AI — Launch or Restart
Exec=bash -c 'cd $HOME/athena && ./athena.sh; exec bash'
Terminal=true
Type=Application
Categories=Utility;
StartupNotify=true
Icon=$HOME/athena/athena-icon.png
```

The `; exec bash` keeps the terminal open after `athena.sh` exits so final output is readable. The `Icon=` path is sed-rewritten by `athena.sh` Phase 2 to the user's actual `$HOME` value before the file gets copied to the desktop.

---

## 19. `README.md`

### Name, location, purpose
- **Name:** `README.md`
- **Location:** Project root → `$HOME/athena/README.md`.
- **Purpose:** User-facing overview. Hardware requirements, install steps, voice command reference, persistence notes, log layout, sampling parameters, file inventory pointer.

### Connection / functions
None. Reference doc. No internal version history — every section describes the current build's behavior.

---

## 20. `FILES.md`

### Name, location, purpose
- **Name:** `FILES.md`
- **Location:** Project root → `$HOME/athena/FILES.md`.
- **Purpose:** Per-file inventory in narrative form. Different audience from this report — `FILES.md` is operator-focused (what does each script do, where do its outputs go), this report is code-review-focused (line numbers, function-by-function, internal mechanics).

---

## 21. `athena-icon-*.png`

- **Names:** `athena-icon-64.png`, `athena-icon-128.png`, `athena-icon-256.png`, `athena-icon.png`.
- **Location:** Project root → copied to `$HOME/athena/`.
- **Purpose:** Desktop launcher icons. The `.desktop` file references only `athena-icon.png`; the sized variants exist for icon-cache systems that prefer specific sizes.

Binary asset files. No code.

---

## 22. Note A — Chat template and Gemma 4 channel tokens

### How the chat template gets selected

`llama-cpp-python`'s `Llama.create_chat_completion()` chooses a chat formatter using this precedence:

1. `chat_handler` parameter, if provided.
2. `chat_format` parameter, if provided (named built-in like `"chatml"`, `"llama-2"`, `"gemma"`, etc.).
3. **`tokenizer.chat_template` from the GGUF metadata** — the embedded jinja2 template packaged into the GGUF when it was converted from the HuggingFace original.
4. Fallback: hardcoded llama-2 chat format.

In `athena_pipeline.py:load_llm()`:

```python
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=LLM_CONFIG["n_ctx"],
    n_gpu_layers=LLM_CONFIG["n_gpu_layers"],
    n_batch=LLM_CONFIG["n_batch"],
    verbose=True,
)
```

No `chat_handler`, no `chat_format`. Precedence falls through to the GGUF embedded template. `Huihui-gemma-4-E2B-it-abliterated-v2.Q4_K_M.gguf` was converted via llama.cpp's `convert_hf_to_gguf.py`, which writes the HF tokenizer's `chat_template` field into the GGUF metadata as `tokenizer.chat_template` — that's what gets used at runtime. The `verbose=True` boot log line `Selected chat format: ...` confirms this path.

### Channel-token format

When `<|think|>` is at the head of the system prompt (per-turn thinking trigger fires), Gemma 4 abliterated emits its reasoning as a structured channel block. The literal output, captured by the `[llm-raw-full]` diagnostic line in `pipeline.log`:

```
<|channel>thought
Thinking Process:
1. **Analyze the request:** ...
2. ...
<channel|>
We are doing very well, Kevin. I am ready to assist you.
```

The opening token is `<|channel>` (pipe-channel-bracket, asymmetric). The closing token is `<channel|>` (bracket-channel-pipe). The text between the two is the thinking; the text after `<channel|>` is the reply. The channel name `thought` appears as the first token inside the open marker — it's metadata indicating which kind of channel block this is. When `<|think|>` is NOT in the system prompt, Gemma emits plain reply text with no channel tokens.

The pipeline does not strip these tokens before saving to conversation history — `full_raw` goes in verbatim. `clean_for_tts` does strip them on the way out to TTS via the `<\|[^|>]*\|?>` and `<[^|<>\s]*\|>` regex passes, so the user only hears the post-`<channel|>` reply text spoken aloud while the thinking block is still preserved in history for future turn references.

### What this enables specifically for Gemma 4

The structured channel block means Gemma's reasoning is preserved in history and available to subsequent turns. If you ask "what was the last thing you thought about?" the model can reference the previous turn's bullet points because they're sitting in `conversation` as part of an assistant message. The trigger is purely the system prompt — having channel tokens in prior assistant messages does not re-fire thinking on subsequent turns. Thinking activates only when the current turn's system prompt has `<|think|>`.

---

## 23. Note B — Memory layout and headroom

### What memory the LLM actually uses

Three buckets, all on GPU:

1. **Model weights** — Gemma 4 E2B Q4_K_M GGUF: ~1.7 GB. Fixed at load time.
2. **KV cache** — sized to `n_ctx = 4096` tokens at load time. For a Gemma 4 E2B-shaped model with the embedded GQA configuration, ~28 KB per token at fp16 → ~115 MB.
3. **Compute scratch** — temporary buffers for matmul intermediates, attention scores, softmax workspace. Tens of MB, ephemeral.

Total LLM GPU footprint: weights + KV cache + scratch ≈ 1.7 + 0.115 + ~0.05 ≈ 1.9 GB.

### What `memory_phases.log` captures

| Phase | When | What it shows |
|---|---|---|
| BEFORE containers | Just before `docker compose up` | Baseline. ~6 GB MemAvailable, ~976 MB GPU lfb (largest contiguous block). Healthy. |
| LLM loading (Phase 2) | After orchestrator container is up and the LLM Llama() call has returned | ~3.6 GB MemAvailable. **GPU lfb collapses to ~8 MB** — the LLM consumed every contiguous block of unified GPU memory. This is normal Jetson behavior. |
| All containers (Phase 3) | After TTS + Whisper containers are also up | ~3.5 GB MemAvailable. Smaller GPU allocations have filled in around the LLM's footprint. |
| warm-idle (in-container) | After the boot greeting plays | ~1.4 GB MemAvailable from the container's perspective (cgroup accounting differs from host — container number is lower because it doesn't count Cached the same way). |
| post-turn (per utterance) | At every speech-end → throttled 15 s | ~700 MB–1.4 GB Available range during a session. Buddy order-10 (4 MB blocks) typically 0 MB during runtime — heavy fragmentation under load is normal and not a failure mode. |

### Operational notes

- **Headroom = MemAvailable, not MemFree.** MemFree only counts pages literally unused; MemAvailable adds reclaimable cache. The honest number is in the "Headroom summary" line at the bottom of every `memory_diag.sh` snapshot.
- **Container MemUsage in `docker stats` overcounts.** On Jetson unified memory, mmap'd model file pages count toward both container Cached AND system Cached/Available. Adding container totals to host Used double-counts.
- **GPU lfb after LLM load is the binding constraint** for any subsequent CUDA allocation. Kokoro's session creation needs ~300 MB contiguous; whisper.cpp's ggml session needs ~80 MB. Both fit in the post-LLM landscape because their allocations come early (during container start) before fragmentation deepens.
- **The 14 × 256 MB pre-launch pressure block** in `athena.sh` Phase 13 evicts non-CUDA pages from the page cache before the LLM allocation, so the LLM lands in clean memory rather than competing with Linux's reluctance to give up file-backed cache.

---

## 24. Note C — The voice router state machine

The router is the largest single subsystem in `athena_pipeline.py` (lines 1300-1900 area). State variables maintained across iterations:

- `current_mode` ∈ `{"conversation", "adult", "translate", "settings"}` or `None` (idle).
- `last_active_mode` — what to restore on wake from idle.
- `is_idle` — gate flag for stop/wake.
- `tts_muted`, `tts_muted_until` — 5-second post-stop mute window.
- `translate_language`, `translate_language_name`, `saved_translate_language*` — current and saved-across-idle translate target.
- `translate_pending_lang`, `translate_pending_count` — auto-lock counter (locks after `translate_lock_consecutive` matching detections).
- `settings_substate` ∈ `{"awaiting_command", "awaiting_voice", "awaiting_voice_speed", "awaiting_translation_speed", "awaiting_translation_voice", "awaiting_audio_output", None}`.
- `settings_previous_mode` — what mode to return to on settings exit.
- `active_output_device`, `active_voice`, `active_voice_speed`, `active_translation_speed`, `translation_gender` — runtime-mutable settings restored from `.athena_runtime.json` and persisted on every change.

### Per-utterance dispatch order

For every transcribed utterance the router checks (in order):

1. **Universal stop** — `athena_any_match(text_lower, STOP_PHRASES)`. Sets idle, mutes TTS for 5 s, saves current mode as `last_active_mode`. If currently in translate, saves `translate_language` for restoration.
2. **Wake from idle** — `is_idle and athena_any_match(text_lower, WAKE_PHRASES)`. Restores `current_mode = last_active_mode`, restores translate state if applicable, re-checks the rest of the utterance against mode-switch phrases (so "okay athena, adult mode" works in one breath).
3. **Idle drop** — anything else while idle is logged and discarded.
4. **Adult entry** — `athena_match(text_lower, ADULT_WAKE) and current_mode == "conversation"`. Wipes history, swaps `conversation[0]` to `adult_system`, sets `current_mode = "adult"`. Strips the wake phrase from `forwarded_text` so any remainder gets forwarded as the first adult-mode prompt.
5. **Adult exit** — `athena_match(text_lower, ADULT_STOP) and current_mode == "adult"`. Wipes history, swaps to `normal_system`, sets `current_mode = "conversation"`. Same strip-and-forward.
6. **Translate entry** — `athena_match(text_lower, TRANSLATE_WAKE) and current_mode == "conversation"`. Initializes `translate_conversation` with the translate system prompt, clears language locks, sets `current_mode = "translate"`. Silent — no immediate LLM call; waiting for a language.
7. **Settings entry** — `athena_match(text_lower, SETTINGS_WAKE) and current_mode in ("conversation", "adult")`. Saves current mode as `settings_previous_mode`, sets `current_mode = "settings"`, plays the menu prompt on the active output.
8. **Settings sub-state** — when `current_mode == "settings"`, the router branches on `settings_substate`. Each option (audio output, voice, voice speed, translation speed, translation voice) has its own awaiting-value sub-state. After each change is applied the menu returns to `awaiting_command` and stays open until "exit settings" / "done".
9. **Translate sub-state** — when `current_mode == "translate"`:
   - If no language is locked yet, check the utterance against `LANGUAGE_MAP`. On match, increment `translate_pending_count`; lock at threshold.
   - `TRANSLATE_UNLOCK_PHRASES` clears the lock without leaving translate mode.
   - `TRANSLATE_STOPS` exits to conversation, mutes TTS 5 s.
   - "athena speak <language>" forces a relock.
   - Otherwise: forward the utterance for translation. Whisper's `language` field is `auto` (Path A) or `en` (Path B) depending on which speaker direction this is.
10. **Active conversation/adult** — anything not consumed by the above branches passes through. Per-turn thinking trigger applied; LLM called via `stream_llm_to_tts` (streaming) or `send_to_llm` (non-streaming fallback).

### Path A vs Path B (translate mode)

- **Path A** = foreign speaker → English on the headset. Triggered by `current_mode == "translate" and translate_language is None` OR by Whisper's auto-detection returning a non-English language. `send_to_whisper` called with `translate=True, language=auto`. Whisper does the translation server-side and returns English text. The English text is sent straight to TTS (no LLM hop) and played on the headset device.
- **Path B** = English speaker → foreign on the wired speaker. Triggered when a target language is locked AND Whisper detected English. The English text is sent to the LLM with the translate system prompt for translation to native script. For non-Latin buckets (chinese, japanese, hindi), the second pass via `ASIAN_OVERRIDE_PROMPT` runs immediately after, replacing the foreign text with English-letter equivalents. The result goes to TTS with `voice=tgt_voice, lang=tgt_lang_code` (which is `en-us` for non-Latin buckets and the native code for Latin ones) and plays on `external_device` (wired USB speaker).

The headset/speaker routing is hard-wired so a two-person interpreter setup just works: the foreign speaker sees Path A's English on the headset; the English speaker sees Path B's foreign on the wired speaker.

### Wake-word fuzzy matching

Every router check that compares against a phrase containing "athena" runs through `athena_match(text_lower, base_phrase)`. The function generates rewrites of the base phrase using each variant in `ATHENA_WAKE_VARIANTS = ["athena", "a theme a", "a theme", "the theme"]` and tests them all as substrings. Bare phrases without "athena" (like "stop translating" alone) are not promoted to wake-status by this — they only fire when paired with a recognized wake-word variant in the same utterance.

---

*End of report.*
