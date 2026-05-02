#!/usr/bin/env python3
"""
Athena Xai v1 — Orchestrator Pipeline
Mic → Silero VAD (CPU) → Whisper STT (GPU) → Gemma 4 E2B LLM (GPU) → Kokoro TTS (GPU) → Speaker

The orchestrator owns the entire conversation loop:
  - Captures mic audio via PortAudio → resamples → Silero VAD on CPU.
  - Once VAD confirms speech, the captured audio is shipped to the Whisper
    container for transcription (GPU).
  - The transcript is matched against the voice router (wake/stop phrases,
    mode-switch phrases, settings menu, translate-mode locks). Anything
    that survives the router is forwarded to the LLM as the user prompt.
  - The LLM (llama-cpp-python loaded directly into this process, sharing
    the same Python interpreter) generates a reply.
  - Reply text → Kokoro TTS HTTP server → WAV bytes → PortAudio sd.play
    on whichever device matches the current mode/path.

Modes:
  conversation  — default. Streaming TTS, talk freely.
  adult         — flirty/companion persona. History wiped on entry/exit.
  translate     — two-person interpreter. Path A (foreign→English) plays
                  on the headset; Path B (English→foreign) plays on the
                  wired USB speaker. Lang-locked on demand.
  settings      — voice, voice speed, translation speed, translation voice
                  gender, audio output sink. Persisted in .athena_runtime.json.
  idle          — muted gate state; resumes last active mode on wake.

Per-turn thinking trigger:
  In conversation or adult mode, if the user utterance contains the whole
  word "think" anywhere in it, the system prompt for that one LLM call gets
  prefixed with "<|think|>" so Gemma emits a visible reasoning block. Next
  turn reverts unless "think" is said again. Translate and idle never swap.

Asian (non-Latin) double-pass for Path B:
  When translating to a target whose path_b_voice_table bucket is
  "chinese", "japanese", or "hindi", the orchestrator runs a second LLM
  pass over the foreign text using ASIAN_OVERRIDE_PROMPT — instructing
  the model to convert the foreign characters to English letters. The
  pass-2 output replaces the original reply for the TTS step. The
  table entries for these buckets carry lang="en-us" and use American
  voices (am_michael / af_heart) directly, so Kokoro/espeak only ever
  sees English-letter text spoken by an American voice on these paths.

LLM:
  Model: Huihui-gemma-4-E2B-it-abliterated-v2.Q4_K_M.gguf
  Sampling: temperature=1.0, top_p=0.95, top_k=64 (Gemma 4 official only).

Audio:
  PortAudio (ALSA backend) for mic IN and TTS OUT. No PulseAudio.
  Mic 48 kHz → resampled to 16 kHz for VAD/Whisper. TTS 24 kHz → resampled
  to device rate (typically 48 kHz) via scipy.signal.resample.

Logs:
  pipeline.log         — primary event log (this module)
  translation.log      — Path A and Path B turn-by-turn
  memory_phases.log    — host + in-container memory snapshots
  orchestrator.log     — full container stdout
"""

import os, sys, time, json, wave, re, logging, io, collections, threading, queue, subprocess, tempfile
import numpy as np
import sounddevice as sd
import scipy.signal
import requests
import onnxruntime as ort

# Jetson unified memory: avoid pinned allocations
os.environ.setdefault('GGML_CUDA_NO_PINNED', '1')

from llama_cpp import Llama

# ═══════════════════════════════════════════════════════════════════════
#  LLM CONFIG — loaded from /app/llm_config.json
# ═══════════════════════════════════════════════════════════════════════

LLM_CONFIG_PATH = os.environ.get("ATHENA_LLM_CONFIG", "/app/llm_config.json")

def load_llm_config():
    """Load LLM config from JSON file with fallback defaults (Gemma 4 E2B tuned)."""
    defaults = {
        "system_prompt": "You are a large language model named Athena and you are a personal assistant and companion to your user Kevin. Keep your responses short and simple and less than three sentences.",
        "adult_prompt": "",
        "translate_prompt": "Translate to {language} only. Do not respond only translate the user prompt.",
        "enable_thinking": False,
        "adult_thinking": False,
        "streaming": True,
        "headset_alsa_pattern": "Composite Device",
        "path_b_voice": "am_michael",
        "path_b_speed": 0.85,
        "external_alsa_pattern": "UACDemoV1",
        "path_b_force_external_device": True,
        "translate_lock_consecutive": 2,
        "tts_max_chars_per_chunk": 300,
        # v62: only Gemma 4's three officially recommended sampling params remain.
        # presence_penalty / repeat_penalty / max_tokens were Qwen 3.5 baggage at
        # no-op values — dropped to stop wasting per-call processing.
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "n_ctx": 4096,
        "n_gpu_layers": 99,
        "n_batch": 128,
    }
    try:
        with open(LLM_CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        # Merge with defaults (config overrides defaults)
        merged = {**defaults, **cfg}
        return merged
    except FileNotFoundError:
        return defaults
    except Exception as e:
        print(f"WARNING: Failed to load {LLM_CONFIG_PATH}: {e} — using defaults")
        return defaults

LLM_CONFIG = load_llm_config()

# Environment overrides (model path stays as env var — it's a mount path)
LLM_MODEL_PATH = os.environ.get("ATHENA_LLM_MODEL", "/models/llm/Huihui-gemma-4-E2B-it-abliterated-v2.Q4_K_M.gguf")
WHISPER_URL = os.environ.get("ATHENA_WHISPER_URL", "http://localhost:8001")
TTS_URL = os.environ.get("ATHENA_TTS_URL", "http://localhost:8002")
LOG_DIR = os.environ.get("ATHENA_LOG_DIR", "/logs")
TTS_VOICE = os.environ.get("ATHENA_TTS_VOICE", "af_sky")
TTS_SPEED = float(os.environ.get("ATHENA_TTS_SPEED", "1.0"))
VAD_THRESHOLD = float(os.environ.get("ATHENA_VAD_THRESHOLD", "0.5"))
SILENCE_TIMEOUT = float(os.environ.get("ATHENA_SILENCE_TIMEOUT", "1.0"))
MIC_RATE = int(os.environ.get("ATHENA_MIC_RATE", "48000"))
WHISPER_RATE = int(os.environ.get("ATHENA_WHISPER_RATE", "16000"))
VAD_MODEL_PATH = os.environ.get("ATHENA_VAD_MODEL", "/models/vad/silero_vad.onnx")
MIN_SPEECH_FRAMES = int(os.environ.get("ATHENA_MIN_SPEECH_FRAMES", "8"))
MIN_SPEECH_DURATION = 0.3
VAD_RATE = 16000
VAD_CHUNK = 512
VAD_CONTEXT = 64  # v5 context window: last 64 samples prepended to each chunk

# v55 #20: persisted Settings selection (audio output, etc.)
RUNTIME_STATE_PATH = os.environ.get("ATHENA_RUNTIME_STATE", "/logs/.athena_runtime.json")

# Conversation sliding window: system prompt + last N messages (N/2 turns)
MAX_CONVERSATION_MESSAGES = 20

# Pre-buffer: ~500ms of audio before VAD confirmation (16 chunks at 32ms each)
PRE_BUFFER_CHUNKS = 16

# ═══════════════════════════════════════════════════════════════════════
#  VOICE ROUTER — modes, wake/stop phrases, language map
# ═══════════════════════════════════════════════════════════════════════
# Universal stop:  "athena stop" / "thank you athena" → IDLE from any mode
# Universal wake:  "hey athena" / "okay athena"       → restore last mode
# Mode-specific stops also exist (translate, adult) for narrower exit.

WAKE_PHRASES = ["hey athena", "okay athena"]
STOP_PHRASES = ["athena stop", "thank you athena"]
TTS_MUTE_DURATION = 5.0  # seconds to suppress TTS after a stop

TRANSLATE_WAKE = "athena translation mode"
TRANSLATE_STOPS = ["athena stop translating", "athena stop translation", "athena stop translate"]

ADULT_WAKE = "athena adult mode"
ADULT_STOP = "athena normal mode"

# v64: per-turn thinking trigger. If the user utterance contains the whole
# word "think" anywhere in it, the THIS turn's LLM call uses a system prompt
# prefixed with "<|think|>" (the model's thinking activator). The next turn
# reverts to the normal system prompt unless "think" is said again. Only
# fires in conversation and adult modes. Translate and idle do not swap
# system prompts.

# v64: Asian-language double-pass for Path B. After the first translate pass,
# Two-pass output for non-Latin script Path B targets (chinese, japanese,
# hindi, etc.). After the existing translate pass, the foreign text gets a
# second LLM pass through this prompt and is routed to TTS as English-letter
# output. The voices for these buckets live in path_b_voice_table directly.
ASIAN_OVERRIDE_PROMPT = "Only phonemize this to English characters. do not respond and do not translate to English."

# v61: Whisper consistently mishears "athena" as "a theme" or "a theme a".
# Every sentinel containing "athena" — at the start, end, or middle — gets
# its match expanded automatically via athena_match(). Whole-phrase
# substring match is preserved: bare "stop translating" / "settings" do NOT
# trigger anything; only when paired with a recognized wake-word variant.
# Add new variants here when observed in transcripts.
ATHENA_WAKE_VARIANTS = ["athena", "a theme a", "a theme", "the theme"]

def athena_match(text_lower, base_phrase):
    """v61: True if any 'athena'-variant rewrite of base_phrase appears in
    text_lower as a substring. text_lower must already be .lower()'d.
    base_phrase that doesn't contain 'athena' falls back to plain `in` check."""
    bp = base_phrase.lower()
    if "athena" not in bp:
        return bp in text_lower
    for v in ATHENA_WAKE_VARIANTS:
        if bp.replace("athena", v) in text_lower:
            return True
    return False

def athena_any_match(text_lower, phrases):
    return any(athena_match(text_lower, p) for p in phrases)

# ── Settings menu (v54) — runtime audio output switching ──
SETTINGS_WAKE = "athena settings"
SETTINGS_AUDIO_PHRASE = "audio output"
SETTINGS_SPEAKER = "speaker"
SETTINGS_HEADPHONES = "headphones"
SETTINGS_EXIT_PHRASES = ["exit settings", "settings exit", "done", "athena exit"]

# v55 #21: clear an auto-locked translate language without leaving translate mode
TRANSLATE_UNLOCK_PHRASES = ["athena unlock language", "athena clear language"]

LANGUAGE_MAP = {
    # Originally covered (30)
    "spanish": "es", "french": "fr", "german": "de", "italian": "it",
    "portuguese": "pt", "russian": "ru", "japanese": "ja", "chinese": "zh",
    "korean": "ko", "arabic": "ar", "hindi": "hi", "turkish": "tr",
    "dutch": "nl", "polish": "pl", "swedish": "sv", "norwegian": "no",
    "danish": "da", "finnish": "fi", "greek": "el", "czech": "cs",
    "romanian": "ro", "hungarian": "hu", "thai": "th", "vietnamese": "vi",
    "indonesian": "id", "malay": "ms", "ukrainian": "uk", "hebrew": "he",
    "persian": "fa", "tagalog": "tl",
    # v60: additional Whisper-supported languages routed via path_b_voice_table
    "afrikaans": "af", "albanian": "sq", "amharic": "am", "armenian": "hy",
    "assamese": "as", "azerbaijani": "az", "bashkir": "ba", "basque": "eu",
    "belarusian": "be", "bengali": "bn", "bosnian": "bs", "breton": "br",
    "bulgarian": "bg", "burmese": "my", "catalan": "ca", "croatian": "hr",
    "estonian": "et", "faroese": "fo", "galician": "gl", "georgian": "ka",
    "gujarati": "gu", "haitian": "ht", "hausa": "ha", "icelandic": "is",
    "kannada": "kn", "kazakh": "kk", "khmer": "km", "kyrgyz": "ky",
    "lao": "lo", "latin": "la", "latvian": "lv", "lingala": "ln",
    "lithuanian": "lt", "luxembourgish": "lb", "macedonian": "mk",
    "malagasy": "mg", "malayalam": "ml", "maltese": "mt", "maori": "mi",
    "marathi": "mr", "mongolian": "mn", "nepali": "ne", "nynorsk": "nn",
    "occitan": "oc", "pashto": "ps", "punjabi": "pa", "sanskrit": "sa",
    "serbian": "sr", "shona": "sn", "sindhi": "sd", "sinhala": "si",
    "slovak": "sk", "slovenian": "sl", "somali": "so", "sundanese": "su",
    "swahili": "sw", "tajik": "tg", "tamil": "ta", "tatar": "tt",
    "telugu": "te", "tibetan": "bo", "turkmen": "tk", "urdu": "ur",
    "uzbek": "uz", "welsh": "cy", "yiddish": "yi", "yoruba": "yo",
}

# ═══════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════

log_file = os.path.join(LOG_DIR, "pipeline.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("athena")

# ═══════════════════════════════════════════════════════════════════════
#  SILERO VAD
# ═══════════════════════════════════════════════════════════════════════

class SileroVAD:
    def __init__(self, model_path=VAD_MODEL_PATH):
        log.info(f"PIPELINE Loading Silero VAD from {model_path}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"VAD model not found: {model_path}")
        available = ort.get_available_providers()
        log.info(f"PIPELINE ORT providers available: {available}")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        log.info("PIPELINE VAD: CPU (forced — saves GPU memory for LLM/Whisper/TTS)")
        t0 = time.time()
        self._session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
        log.info(f"PIPELINE VAD loaded in {time.time()-t0:.2f}s, active: {self._session.get_providers()}")
        input_names = {i.name: i.shape for i in self._session.get_inputs()}
        log.info(f"PIPELINE VAD inputs: {input_names}")
        self._sr = np.array(VAD_RATE, dtype=np.int64)
        if "state" in input_names:
            self._version = "v5"
            self._state_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_names["state"]]
            self._state = np.zeros(self._state_shape, dtype=np.float32)
            log.info(f"PIPELINE VAD v5/v6: state {self._state_shape}")
        elif "h" in input_names:
            self._version = "v4"
            self._h = np.zeros([d if isinstance(d, int) and d > 0 else 1 for d in input_names["h"]], dtype=np.float32)
            self._c = np.zeros([d if isinstance(d, int) and d > 0 else 1 for d in input_names["c"]], dtype=np.float32)
            log.info("PIPELINE VAD v4")
        else:
            raise RuntimeError(f"Unknown VAD format: {list(input_names.keys())}")
        self._context = np.zeros((1, VAD_CONTEXT), dtype=np.float32)
        log.info(f"PIPELINE VAD context: {VAD_CONTEXT} samples prepended to each {VAD_CHUNK}-sample chunk")
        log.info("PIPELINE VAD warm-up...")
        t0 = time.time()
        self._run_inference(np.zeros((1, VAD_CONTEXT + VAD_CHUNK), dtype=np.float32))
        log.info(f"PIPELINE VAD warm-up: {(time.time()-t0)*1000:.1f}ms")
        self.reset()
        log.info("PIPELINE VAD ready")

    def _run_inference(self, chunk):
        if self._version == "v5":
            out = self._session.run(None, {"input": chunk, "sr": self._sr, "state": self._state})
            self._state = out[1]
        else:
            out = self._session.run(None, {"input": chunk, "sr": self._sr, "h": self._h, "c": self._c})
            self._h, self._c = out[1], out[2]
        return float(out[0].flatten()[0])

    def process_chunk(self, audio_16k):
        if len(audio_16k) != VAD_CHUNK: return 0.0
        try:
            chunk = audio_16k.reshape(1, -1).astype(np.float32)
            input_with_context = np.concatenate([self._context, chunk], axis=1)
            confidence = self._run_inference(input_with_context)
            self._context = chunk[:, -VAD_CONTEXT:]
            return confidence
        except Exception as e:
            log.error(f"PIPELINE VAD error: {e}")
            return 0.0

    def reset(self):
        if self._version == "v5": self._state = np.zeros(self._state_shape, dtype=np.float32)
        else: self._h, self._c = np.zeros_like(self._h), np.zeros_like(self._c)
        self._context = np.zeros((1, VAD_CONTEXT), dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════════
#  TEXT PROCESSING
# ═══════════════════════════════════════════════════════════════════════

# ── Number-to-words helpers (for natural TTS pronunciation) ──

def _number_group_to_words(group_str):
    """Convert a 1-3 digit group to words. e.g. '280' → 'two hundred eighty'."""
    ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
            'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
            'seventeen', 'eighteen', 'nineteen']
    tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
    n = int(group_str)
    if n == 0:
        return ''
    parts = []
    if n >= 100:
        parts.append(ones[n // 100] + ' hundred')
        n %= 100
    if n >= 20:
        w = tens[n // 10]
        if n % 10:
            w += ' ' + ones[n % 10]
        parts.append(w)
    elif n > 0:
        parts.append(ones[n])
    return ' '.join(parts)

def _expand_comma_number(match):
    """Expand 5,280 → 'five thousand two hundred eighty'. Up to trillions."""
    text = match.group(0)
    groups = text.split(',')
    num_groups = len(groups)
    scales = ['', ' thousand', ' million', ' billion', ' trillion']
    if num_groups > len(scales):
        return text
    parts = []
    for i, group in enumerate(groups):
        scale_idx = num_groups - 1 - i
        words = _number_group_to_words(group)
        if words:
            parts.append(words + scales[scale_idx])
    return ' '.join(parts) if parts else 'zero'

def _expand_decimal(match):
    """Expand 3.14 → 'three point one four'."""
    whole = match.group(1)
    decimal = match.group(2)
    try:
        whole_int = int(whole)
        whole_words = 'zero' if whole_int == 0 else _number_group_to_words(whole)
    except ValueError:
        whole_words = whole
    digit_words = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                   '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
    decimal_words = ' '.join(digit_words.get(d, d) for d in decimal)
    return f"{whole_words} point {decimal_words}"

def numbers_to_words(text):
    """Convert numbers in text to spoken-word form so TTS doesn't sound stupid.
    Handles decimals (3.14) and comma-separated numbers (5,280) up to trillions."""
    text = re.sub(r'(\d+)\.(\d+)(?!\.)', _expand_decimal, text)
    text = re.sub(r'\d{1,3}(?:,\d{3})+', _expand_comma_number, text)
    return text

def split_for_tts(text, max_chars):
    """v55 #23: split a long string into <=max_chars pieces at sentence-ish
    boundaries to keep Kokoro inputs small and avoid the NvMap allocator
    burst (`NvMapMemAllocInternalTagged: 1075072515 error 12`) that degrades
    RTF on paragraph-sized chunks. Splits prefer sentence punctuation, then
    commas, then whitespace, then hard cuts."""
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return [text] if text else []
    out = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = -1
        for sep in [". ", "! ", "? ", "; ", ": ", ", ", " "]:
            idx = window.rfind(sep)
            if idx > max_chars // 2:  # prefer cuts in second half of window
                cut = idx + len(sep)
                break
        if cut < 0:
            cut = max_chars
        piece = remaining[:cut].strip()
        if piece:
            out.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        out.append(remaining)
    return out

def clean_for_tts(raw):
    """Strip markdown, special tokens; expand numbers.
    v63: extract_thinking and the streaming gate were removed. clean_for_tts
    no longer pre-strips thinking — the full LLM output flows through.
    v61: removed the emoji-strip regex. Its U+24C2-U+1F251 range was
    swallowing all Hiragana / Katakana / CJK Unified / CJK Symbols, which
    silently killed Japanese / Chinese / Korean Path B output."""
    c = raw
    for pat, rep in [(r'\*\*(.+?)\*\*', r'\1'), (r'\*(.+?)\*', r'\1'), (r'__(.+?)__', r'\1'),
                     (r'^#{1,6}\s*', ''), (r'```[\s\S]*?```', ''), (r'`(.+?)`', r'\1'),
                     (r'^[\s]*[-*+]\s+', ''), (r'^[\s]*\d+\.\s+', ''), (r'\(.*?\)', ''), (r'\[.*?\]', '')]:
        c = re.sub(pat, rep, c, flags=re.MULTILINE)
    # Strip any residual Gemma 4 special tokens
    c = re.sub(r'<\|[^|>]*\|?>', '', c)
    c = re.sub(r'<[^|<>\s]*\|>', '', c)

    # v59 punctuation normalization. Replace with spaces (or word equivalents)
    # so the cleaner doesn't fuse adjacent words (e.g. "Size/Volume" → "Size Volume")
    # and Kokoro doesn't get tripped up trying to voice symbols.
    c = c.replace('=', ' equals ')
    c = c.replace('+', ' plus ')
    c = re.sub(r'[\*/\\_]', ' ', c)   # orphan asterisks, slashes, backslashes, underscores → space

    c = re.sub(r'\n+', ' ', c)
    c = re.sub(r'\s+', ' ', c).strip()
    c = numbers_to_words(c)
    return c

# ═══════════════════════════════════════════════════════════════════════
#  AUDIO
# ═══════════════════════════════════════════════════════════════════════

def resample_audio(audio, from_rate, to_rate):
    if from_rate == to_rate: return audio
    return scipy.signal.resample(audio, int(len(audio) * to_rate / from_rate)).astype(np.float32)

def audio_to_wav_bytes(audio_f32, sr):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(np.clip(audio_f32 * 32767, -32768, 32767).astype(np.int16).tobytes())
    buf.seek(0)
    return buf.read()

# ═══════════════════════════════════════════════════════════════════════
#  SERVICE CALLS
# ═══════════════════════════════════════════════════════════════════════

def send_to_whisper(wav_bytes, language_override=None, translate_override=None, duration_hint=None):
    """Send audio to Whisper. Returns (text, detected_lang).
    v58: empty language_override now sends `language=auto` to whisper-server.
    Previously we omitted the form parameter, which made whisper-server fall
    back to its default of `lang=en` — so auto-detect never fired and Path A
    was effectively dead. With `language=auto` whisper.cpp actually detects.
    v55 #17: timeout scales with duration_hint (seconds of audio).
    language_override="" + translate_override=True = Whisper translates the
    detected source language to English server-side. Multilingual model required."""
    lang = language_override if language_override is not None else ""
    do_translate = translate_override if translate_override is not None else False
    timeout = 30 if duration_hint is None else max(30, int(duration_hint * 2) + 15)
    # v58: empty string → "auto" so whisper-server actually detects.
    lang_for_request = lang if lang else "auto"
    log.info(f"PIPELINE >>> Whisper: {len(wav_bytes)} bytes (lang={lang_for_request!r}, translate={do_translate}, timeout={timeout}s)")
    t0 = time.time()
    try:
        form_data = {"response_format": "verbose_json", "temperature": "0.0",
                     "language": lang_for_request}
        if do_translate:
            form_data["translate"] = "true"
        r = requests.post(f"{WHISPER_URL}/inference",
                          files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                          data=form_data, timeout=timeout)
        if r.status_code == 200:
            result = r.json()
            text = result.get("text", "").strip()
            detected_lang = result.get("detected_language", "") or result.get("language", "")
            log.info(f"PIPELINE <<< Whisper ({time.time()-t0:.2f}s): '{text}' lang={detected_lang!r}")
            return text, detected_lang
        log.error(f"PIPELINE Whisper HTTP {r.status_code}")
    except Exception as e: log.error(f"PIPELINE Whisper: {e}")
    return "", ""

def load_llm():
    """LLM Charge-up: load model directly into GPU memory."""
    log.info("=" * 50)
    log.info("  LLM Charge-up — Loading into GPU")
    log.info("=" * 50)
    log.info(f"  Model:      {LLM_MODEL_PATH}")
    log.info(f"  n_ctx:      {LLM_CONFIG['n_ctx']}")
    log.info(f"  n_gpu_layers: {LLM_CONFIG['n_gpu_layers']}")
    log.info(f"  n_batch:    {LLM_CONFIG['n_batch']}")
    log.info(f"  GGML_CUDA_NO_PINNED: {os.environ.get('GGML_CUDA_NO_PINNED', 'not set')}")
    t0 = time.time()
    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=LLM_CONFIG["n_ctx"],
        n_gpu_layers=LLM_CONFIG["n_gpu_layers"],
        n_batch=LLM_CONFIG["n_batch"],
        verbose=True,
    )
    elapsed = time.time() - t0
    log.info(f"  LLM loaded in {elapsed:.1f}s")
    log.info("=" * 50)
    return llm

_NUMBER_WORDS = {"zero":0, "one":1, "two":2, "three":3, "four":4, "five":5,
                 "six":6, "seven":7, "eight":8, "nine":9, "ten":10}

def _parse_speed_step(text_lower):
    """v60: extract a 1–9 integer step from a Whisper transcript. Accepts
    digit form ("3"), word form ("three"), or padded ("step three.").
    Returns int or None."""
    m = re.search(r'\b([1-9])\b', text_lower)
    if m:
        return int(m.group(1))
    for word, n in _NUMBER_WORDS.items():
        if word in text_lower:
            if 1 <= n <= 9:
                return n
    return None

def send_to_llm(text, conversation, llm):
    """v64: thinking activation is per-turn — see the in-router system-prompt
    swap in main() that detects the whole word 'think' in the user utterance
    and prefixes conversation[0] with '<|think|>' for that turn only."""
    conversation.append({"role": "user", "content": text})
    log.info(f"PIPELINE >>> LLM: '{text[:80]}'")
    t0 = time.time()
    # v62: Gemma 4's three official params only.
    call_kwargs = dict(
        messages=conversation,
        temperature=LLM_CONFIG["temperature"],
        top_p=LLM_CONFIG["top_p"],
        top_k=LLM_CONFIG["top_k"],
    )
    try:
        response = llm.create_chat_completion(**call_kwargs)
        elapsed = time.time() - t0
        msg = response["choices"][0]["message"]
        reply = msg.get("content", "") or ""
        finish = response["choices"][0].get("finish_reason", "?")
        usage = response.get("usage", {})
        prompt_tok = usage.get("prompt_tokens", 0)
        completion_tok = usage.get("completion_tokens", 0)
        tok_s = completion_tok / elapsed if elapsed > 0 else 0

        reply_words = len(reply.split()) if reply else 0

        # Tok/s report — front and center
        log.info(f"PIPELINE <<< LLM ({elapsed:.2f}s)")
        log.info(f"  Prompt:   {prompt_tok} tokens")
        log.info(f"  Reply:    {completion_tok} tokens @ {tok_s:.1f} tok/s")
        log.info(f"  Reply:    {reply_words} words")
        log.info(f"  Finish:   {finish}")
        log.info(f"  Text:     '{reply[:80]}'")

        # v63: diagnostic capture for raw LLM output (special tokens visible).
        log.info(f"PIPELINE [llm-raw-full] bytes={len(reply)} :: {reply!r}")

        # Log raw for debugging
        raw_str = json.dumps({"content": reply[:200], "finish_reason": finish,
                              "prompt_tokens": prompt_tok, "completion_tokens": completion_tok,
                              "tok_s": round(tok_s, 1)})
        log.info(f"PIPELINE [llm-raw] {raw_str}")

        # v63: save full reply verbatim to history. No extraction, no branching.
        conversation.append({"role": "assistant", "content": reply})
        # Sliding window: keep system prompt at [0] + last N messages
        if len(conversation) > MAX_CONVERSATION_MESSAGES + 1:
            del conversation[1:len(conversation) - MAX_CONVERSATION_MESSAGES]
        return reply
    except Exception as e:
        log.error(f"PIPELINE LLM: {e}", exc_info=True)
    # On any failure, store placeholder so conversation doesn't have consecutive user messages
    conversation.append({"role": "assistant", "content": "..."})
    if len(conversation) > MAX_CONVERSATION_MESSAGES + 1:
        del conversation[1:len(conversation) - MAX_CONVERSATION_MESSAGES]
    log.warning("PIPELINE LLM failed, stored placeholder in conversation")
    return ""

def send_to_tts(text, voice=None, speed=None, lang=None):
    """Send text to Kokoro TTS. Optional voice + speed + lang overrides for
    per-mode use. v60 added lang for Path B foreign-language phonemization."""
    use_voice = voice if voice is not None else TTS_VOICE
    use_speed = speed if speed is not None else TTS_SPEED
    use_lang = lang if lang is not None else "en-us"
    log.info(f"PIPELINE >>> TTS: '{text[:60]}' voice={use_voice} speed={use_speed} lang={use_lang}")
    t0 = time.time()
    try:
        r = requests.post(f"{TTS_URL}/synthesize",
                          json={"text": text, "voice": use_voice, "speed": use_speed, "lang": use_lang}, timeout=60)
        if r.status_code == 200:
            log.info(f"PIPELINE <<< TTS: {len(r.content)} bytes ({time.time()-t0:.2f}s)")
            return r.content
        log.error(f"PIPELINE TTS HTTP {r.status_code}")
    except Exception as e: log.error(f"PIPELINE TTS: {e}")
    return None

# v57: PulseAudio is gone. Output goes through PortAudio→ALSA via /dev/snd
# (the same path the mic uses). No paplay, no socket, no cookie, no auth.
# Routing is by PortAudio device INDEX, not by PA sink name. Indexes are
# discovered at startup with sd.query_devices() — same mechanism the mic uses.
# sd.stop() actually works again because output is back on sounddevice.

def kill_active_paplay():
    """v57: now just sd.stop(). Name kept so existing call sites don't change.
    sounddevice's sd.stop() cancels any in-flight sd.play()."""
    try:
        sd.stop()
    except Exception as e:
        log.warning(f"PIPELINE sd.stop: {e}")

def load_runtime_state():
    """v55 #20: read persisted Settings selection (audio output, etc.)
    from RUNTIME_STATE_PATH. Returns dict; empty on missing/corrupt file."""
    try:
        with open(RUNTIME_STATE_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except FileNotFoundError:
        return {}
    except Exception as e:
        log.warning(f"PIPELINE load_runtime_state: {e}")
    return {}

def save_runtime_state(state):
    """v55 #20: persist Settings selection so it survives pipeline restarts."""
    try:
        os.makedirs(os.path.dirname(RUNTIME_STATE_PATH), exist_ok=True)
        with open(RUNTIME_STATE_PATH, "w") as f:
            json.dump(state, f)
    except Exception as e:
        log.warning(f"PIPELINE save_runtime_state: {e}")

def find_pa_device(pattern, kind="input"):
    """Find a PortAudio device INDEX by case-insensitive name substring match.
    Used for mic input only in v54 (output goes through paplay → PA sinks).
    kind='input' filters to devices with input channels; 'output' for speakers.
    Returns the integer index or None."""
    if not pattern:
        return None
    try:
        for i, dev in enumerate(sd.query_devices()):
            name = dev.get('name', '')
            if pattern.lower() not in name.lower():
                continue
            if kind == "output" and dev.get('max_output_channels', 0) > 0:
                return i
            if kind == "input" and dev.get('max_input_channels', 0) > 0:
                return i
    except Exception as e:
        log.warning(f"PIPELINE find_pa_device({pattern!r}, {kind!r}): {e}")
    return None

def resolve_path_b_voice(target_language, gender):
    """v60: look up the Kokoro lang code + voice for Path B's target language.
    target_language is a lowercase spoken-name string (e.g., 'spanish', 'korean',
    'german'). Falls back to the `_fallback` row when the language isn't in
    path_b_voice_table. gender is a single string ('male' or 'female') applied
    uniformly across all language buckets — one global setting.
    Returns (lang_code, voice_id, bucket_name)."""
    table = LLM_CONFIG.get("path_b_voice_table", {})
    fallback = table.get("_fallback", {"lang": "en-us", "female": "af_sky",
                                       "male": "am_michael", "bucket": "fallback"})
    row = table.get((target_language or "").lower(), fallback)
    bucket = row.get("bucket", "fallback")
    g = (gender or "male").lower()
    voice = row.get(g) or row.get("male") or row.get("female")
    return row.get("lang", "en-us"), voice, bucket

# v60: separate translation log for review of every translation turn.
TRANSLATION_LOG_PATH = os.path.join(LOG_DIR, "translation.log")
translation_log = logging.getLogger("athena.translation")
translation_log.addHandler(logging.FileHandler(TRANSLATION_LOG_PATH))
translation_log.setLevel(logging.INFO)

def play_to_device(wav_bytes, device_idx):
    """v57: play WAV bytes to a PortAudio output device by INDEX, via sd.play.
    Goes PortAudio → ALSA → /dev/snd. No PulseAudio. Resamples client-side
    from Kokoro's 24 kHz to the device's native rate using scipy. Blocking."""
    if not wav_bytes:
        return 0.0
    if device_idx is None:
        log.error("PIPELINE play_to_device: no device_idx (LOCKED, refusing to play)")
        return 0.0
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            raw = wf.readframes(n_frames)
        duration = n_frames / sr if sr else 0.0
        # Decode int16 PCM to float32 in [-1, 1]
        if sw != 2:
            log.error(f"PIPELINE play_to_device: unexpected sample width {sw}")
            return 0.0
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_ch > 1:
            audio = audio.reshape(-1, n_ch)
        # Resample to the device's native rate
        try:
            dev_info = sd.query_devices(device_idx)
            device_rate = int(dev_info.get('default_samplerate') or sr)
        except Exception:
            device_rate = sr
        if sr != device_rate:
            audio = resample_audio(audio, sr, device_rate)
        log.info(f"PIPELINE [audio-monitor] sd.play device={device_idx} src={sr}Hz "
                 f"→ {device_rate}Hz ch={n_ch} duration={duration:.2f}s")
        sd.play(audio, samplerate=device_rate, device=device_idx)
        sd.wait()
        return duration
    except Exception as e:
        log.error(f"PIPELINE play_to_device: {e}")
        return 0.0

# ═══════════════════════════════════════════════════════════════════════
#  STREAMING PIPELINE — LLM → sentence buffer → TTS → audio queue
# ═══════════════════════════════════════════════════════════════════════
# 3 concurrent stages:
#   1. LLM thread (this function): iterates llama-cpp-python create_chat_completion(stream=True),
#      accumulates tokens, splits on '\n' into paragraph-sized chunks, pushes to sentence_q.
#   2. TTS dispatcher thread: pulls from sentence_q, calls clean_for_tts + send_to_tts,
#      pushes resulting WAV bytes to audio_q. Calls on_first_chunk callback when first
#      chunk lands in audio_q (used to resume mic listening early).
#   3. Audio player thread: pulls from audio_q, calls play_to_sink (paplay → locked
#      PA sink), serializing playback in arrival order. PA does rate conversion.
# Sentinels (None) propagate through both queues to signal completion.
# Sentence delimiter is '\n' only — periods/colons/question marks flow through.
# Channel-block thinking is gated: chunks are not emitted while inside an unclosed <|channel.

def stream_llm_to_tts(text, conversation, llm, device_idx, voice=None, speed=None, lang=None, on_first_chunk=None):
    """Stream LLM tokens through sentence buffer to TTS to audio playback.
    v57: device_idx is a PortAudio output device INDEX (not a PA sink name).
         Audio plays via sd.play() → PortAudio → ALSA directly.
    v60: lang passed to TTS so Path B can render foreign output through the
         correct phonemizer.
    v62: per-turn THINK trigger and chat_template_kwargs path are gone.
         Thinking activation is mode-driven via system-prompt swap.
    Returns (reply_text, elapsed_seconds). Blocks until all audio finishes."""
    conversation.append({"role": "user", "content": text})
    log.info(f"PIPELINE >>> LLM (stream): '{text[:80]}' device={device_idx}")
    t0 = time.time()

    use_voice = voice if voice is not None else TTS_VOICE
    use_speed = speed if speed is not None else TTS_SPEED
    use_lang = lang if lang is not None else "en-us"
    max_chars = int(LLM_CONFIG.get("tts_max_chars_per_chunk", 300))

    sentence_q = queue.Queue()
    audio_q = queue.Queue()
    chunks_sent = [0]

    def tts_worker():
        try:
            while True:
                sentence = sentence_q.get()
                if sentence is None:
                    break
                try:
                    cleaned = clean_for_tts(sentence)
                except Exception as e:
                    log.error(f"PIPELINE clean_for_tts: {e}")
                    continue
                if not cleaned:
                    continue
                for piece in split_for_tts(cleaned, max_chars):
                    if not piece:
                        continue
                    try:
                        tts_audio = send_to_tts(piece, voice=use_voice, speed=use_speed, lang=use_lang)
                    except Exception as e:
                        log.error(f"PIPELINE send_to_tts: {e}")
                        tts_audio = None
                    if tts_audio:
                        audio_q.put(tts_audio)
                        chunks_sent[0] += 1
                        log.info(f"PIPELINE Stream chunk {chunks_sent[0]}: '{piece[:60]}'")
                        if chunks_sent[0] == 1 and on_first_chunk:
                            try:
                                on_first_chunk()
                            except Exception as e:
                                log.error(f"PIPELINE on_first_chunk callback: {e}")
        except Exception as e:
            log.error(f"PIPELINE tts_worker fatal: {e}", exc_info=True)
        finally:
            audio_q.put(None)

    def audio_player():
        try:
            while True:
                wav_bytes = audio_q.get()
                if wav_bytes is None:
                    break
                try:
                    play_to_device(wav_bytes, device_idx)
                except Exception as e:
                    log.error(f"PIPELINE audio_player: {e}")
        except Exception as e:
            log.error(f"PIPELINE audio_player fatal: {e}", exc_info=True)

    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    player_thread = threading.Thread(target=audio_player, daemon=True)
    tts_thread.start()
    player_thread.start()

    full_raw = ""
    reply_buffer = ""
    finish_reason = "?"
    prompt_tokens = 0
    completion_tokens = 0

    # v63: streaming gate removed. Full LLM output streams to TTS verbatim;
    # full_raw is also saved to conversation history verbatim.

    # v62: Gemma 4's three official params only.
    stream_kwargs = dict(
        messages=conversation,
        temperature=LLM_CONFIG["temperature"],
        top_p=LLM_CONFIG["top_p"],
        top_k=LLM_CONFIG["top_k"],
        stream=True,
    )
    try:
        stream = llm.create_chat_completion(**stream_kwargs)

        for chunk in stream:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {}).get("content", "") or ""
            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr
            usage = chunk.get("usage") or {}
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
            if not delta:
                continue
            full_raw += delta
            reply_buffer += delta

            # v63: streaming gate removed. All content flows to TTS verbatim.

            # \n-delimited paragraph chunks (periods, etc. flow through)
            if '\n' in reply_buffer:
                parts = reply_buffer.split('\n')
                for sentence in parts[:-1]:
                    if sentence.strip():
                        sentence_q.put(sentence)
                reply_buffer = parts[-1]

        # Flush trailing content as final chunk
        if reply_buffer.strip():
            sentence_q.put(reply_buffer)

        # Signal workers and wait
        sentence_q.put(None)
        tts_thread.join()
        player_thread.join()

    except Exception as e:
        log.error(f"PIPELINE Stream error: {e}", exc_info=True)
        try:
            sentence_q.put(None)
            tts_thread.join(timeout=5)
            player_thread.join(timeout=5)
        except Exception:
            pass

    # v57: nothing to clean up — sd.play is per-chunk and self-contained.
    elapsed = time.time() - t0
    # v63: extract_thinking removed. Reply is full_raw verbatim.
    reply_text = full_raw
    # v61 #4: llama-cpp-python's streaming usage block isn't always populated
    # per-chunk; we'd often log "0 tokens @ 0.0 tok/s" even when the LLM
    # produced a real reply. Approximate from char/word counts when usage
    # comes back empty.
    if not completion_tokens and full_raw:
        approx_words = len(full_raw.split())
        approx_tokens = max(1, len(full_raw) // 4)  # rough en-text heuristic
        tok_s = (approx_tokens / elapsed) if elapsed > 0 else 0
        token_label = f"~{approx_tokens} tokens (~{approx_words} words, est)"
    else:
        approx_tokens = completion_tokens
        tok_s = (completion_tokens / elapsed) if (elapsed > 0 and completion_tokens) else 0
        token_label = f"{completion_tokens} tokens"

    log.info(f"PIPELINE <<< LLM stream ({elapsed:.2f}s)")
    log.info(f"  Prompt:   {prompt_tokens} tokens")
    log.info(f"  Reply:    {token_label} @ {tok_s:.1f} tok/s")
    log.info(f"  Chunks:   {chunks_sent[0]}  Finish: {finish_reason}")
    log.info(f"  Text:     '{reply_text[:80]}'")

    # v63: diagnostic capture of raw stream output (special tokens visible).
    log.info(f"PIPELINE [llm-raw-full] bytes={len(full_raw)} :: {full_raw!r}")

    # v63: save full_raw verbatim. No branching, no extraction.
    conversation.append({"role": "assistant", "content": full_raw})
    if len(conversation) > MAX_CONVERSATION_MESSAGES + 1:
        del conversation[1:len(conversation) - MAX_CONVERSATION_MESSAGES]

    return reply_text, elapsed

MEMORY_PHASES_LOG = os.path.join(LOG_DIR, "memory_phases.log")
_last_mem_snapshot_time = 0.0

def memory_snapshot(label, throttle_seconds=0):
    """v62: write an in-container memory snapshot to memory_phases.log.
    Container-eye view (via /proc) — host-side memory_diag.sh from athena.sh
    has the fuller picture (tegrastats, nvmap, docker stats). This sidecar
    captures warm-idle and per-turn moments the host snapshots can't see.

    throttle_seconds: if >0, skip the snapshot when we wrote one less than
    that many seconds ago. Useful for per-turn snapshots so a fast-fire
    session doesn't fill the log."""
    global _last_mem_snapshot_time
    now = time.time()
    if throttle_seconds > 0 and (now - _last_mem_snapshot_time) < throttle_seconds:
        return
    _last_mem_snapshot_time = now
    try:
        meminfo = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                k, _, v = line.partition(":")
                meminfo[k.strip()] = v.strip()

        # buddyinfo: order-10 (4MB) sum across zones — Jetson's CUDA-allocatable
        # contiguous proxy. Not as precise as tegrastats lfb but it's what we
        # can see from inside the container.
        order10_total_4mb = 0
        zone_lines = []
        try:
            with open("/proc/buddyinfo", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 14:
                        # Node N, zone NAME, then 11 order columns (0..10)
                        zone = parts[3]
                        try:
                            o10 = int(parts[14])
                        except (IndexError, ValueError):
                            o10 = 0
                        order10_total_4mb += o10
                        zone_lines.append(f"{zone}={o10}x4MB")
        except FileNotFoundError:
            zone_lines.append("(buddyinfo unavailable)")

        avail_kb = int(meminfo.get("MemAvailable", "0 kB").split()[0] or 0)
        free_kb  = int(meminfo.get("MemFree",      "0 kB").split()[0] or 0)
        cached_kb = int(meminfo.get("Cached",      "0 kB").split()[0] or 0)
        anon_kb   = int(meminfo.get("AnonPages",   "0 kB").split()[0] or 0)
        slab_kb   = int(meminfo.get("Slab",        "0 kB").split()[0] or 0)
        total_kb  = int(meminfo.get("MemTotal",    "0 kB").split()[0] or 0)

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(MEMORY_PHASES_LOG, "a") as f:
            f.write(f"  ── In-container snapshot: {label} @ {ts} ──\n")
            f.write(f"    MemTotal={total_kb//1024}MB  MemAvailable={avail_kb//1024}MB  "
                    f"MemFree={free_kb//1024}MB  Cached={cached_kb//1024}MB  "
                    f"AnonPages={anon_kb//1024}MB  Slab={slab_kb//1024}MB\n")
            f.write(f"    buddy order-10: total={order10_total_4mb*4}MB across zones [{', '.join(zone_lines)}]\n")
            f.write("\n")
    except Exception as e:
        log.warning(f"PIPELINE memory_snapshot({label}) failed: {e}")

def list_audio_devices():
    """Log every PortAudio device PortAudio sees. This is the discovery surface
    used for lock-by-name matching."""
    log.info("PIPELINE PortAudio devices (sd.query_devices):")
    try:
        for i, d in enumerate(sd.query_devices()):
            m = ""
            try:
                if i == sd.default.device[0]: m += " [IN]"
                if i == sd.default.device[1]: m += " [OUT]"
            except Exception:
                pass
            log.info(f"  {i}: {d['name']} (in:{d['max_input_channels']} out:{d['max_output_channels']}){m}")
    except Exception as e:
        log.error(f"PIPELINE Audio query: {e}")

def wait_for_services():
    """Wait for TTS and Whisper to come online. They start after LLM loads."""
    log.info("PIPELINE Waiting for TTS + Whisper services...")

    tts_ok = False
    for i in range(90):  # 90 x 2s = 3 minutes max
        try:
            r = requests.get(f"{TTS_URL}/health", timeout=2)
            if r.status_code == 200:
                tts_ok = True
                break
        except Exception:
            pass
        if (i + 1) % 10 == 0:
            log.info(f"PIPELINE   TTS: waiting ({(i+1)*2}s)...")
        time.sleep(2)
    if tts_ok:
        log.info("PIPELINE   TTS: ONLINE")
    else:
        log.warning("PIPELINE   TTS: FAILED (continuing without TTS)")

    whisper_ok = False
    for i in range(60):  # 60 x 2s = 2 minutes max
        try:
            r = requests.get(f"{WHISPER_URL}/health", timeout=2)
            if r.status_code == 200:
                whisper_ok = True
                break
        except Exception:
            pass
        if (i + 1) % 10 == 0:
            log.info(f"PIPELINE   Whisper: waiting ({(i+1)*2}s)...")
        time.sleep(2)
    if whisper_ok:
        log.info("PIPELINE   Whisper: ONLINE")
    else:
        log.warning("PIPELINE   Whisper: FAILED (continuing without STT)")

    return tts_ok, whisper_ok

# ═══════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("  Athena Xai v1 — Pipeline Starting")
    log.info("=" * 60)
    log.info(f"  LLM:     direct (llama-cpp-python)")
    log.info(f"  Model:   {LLM_MODEL_PATH}")
    log.info(f"  Config:  {LLM_CONFIG_PATH}")
    log.info(f"  Whisper: {WHISPER_URL}")
    log.info(f"  TTS:     {TTS_URL}")
    log.info(f"  Voice:   {TTS_VOICE} @ {TTS_SPEED}x")
    log.info(f"  VAD:     threshold={VAD_THRESHOLD}, silence={SILENCE_TIMEOUT}s, "
             f"confirm={MIN_SPEECH_FRAMES} frames ({MIN_SPEECH_FRAMES*32}ms)")
    log.info(f"  Mic:     {MIC_RATE}Hz → VAD {VAD_RATE}Hz → Whisper {WHISPER_RATE}Hz")
    log.info(f"  Pre-buffer: {PRE_BUFFER_CHUNKS} chunks (~{PRE_BUFFER_CHUNKS*32}ms)")
    log.info(f"  Conversation: sliding window, system + last {MAX_CONVERSATION_MESSAGES} messages")
    log.info(f"  LLM params: temp={LLM_CONFIG['temperature']} top_p={LLM_CONFIG['top_p']} "
             f"top_k={LLM_CONFIG['top_k']} (Gemma 4 official)")
    log.info(f"  LLM model: n_ctx={LLM_CONFIG['n_ctx']} n_gpu_layers={LLM_CONFIG['n_gpu_layers']} "
             f"n_batch={LLM_CONFIG['n_batch']}")
    log.info("")
    list_audio_devices()

    vad = SileroVAD(model_path=VAD_MODEL_PATH)

    # v47: LLM loads FIRST into clean contiguous memory.
    # TTS+Whisper start AFTER via athena.sh — smaller allocations fill gaps.
    llm = load_llm()

    # Build mode-specific system prompts. Gemma 4 thinking is activated by prefixing
    # the literal token "<|think|>" to the system prompt content. Per-mode toggles.
    enable_thinking = bool(LLM_CONFIG.get("enable_thinking", False))
    adult_thinking = bool(LLM_CONFIG.get("adult_thinking", True))
    streaming_enabled = bool(LLM_CONFIG.get("streaming", True))

    def _wrap_thinking(prompt, on):
        return ("<|think|>" + prompt) if on else prompt

    normal_system = _wrap_thinking(LLM_CONFIG["system_prompt"], enable_thinking)
    adult_base = LLM_CONFIG.get("adult_prompt", "") or LLM_CONFIG["system_prompt"]
    adult_system = _wrap_thinking(adult_base, adult_thinking)
    conversation = [{"role": "system", "content": normal_system}]
    log.info(f"PIPELINE System prompt: '{LLM_CONFIG['system_prompt'][:80]}'")
    log.info(f"PIPELINE Adult prompt:  '{adult_base[:80]}'")
    log.info(f"PIPELINE Thinking:      normal={enable_thinking} adult={adult_thinking} per-turn=word'think'")
    log.info(f"PIPELINE Streaming:     {streaming_enabled}")
    log.info(f"PIPELINE Voice router:  wake={WAKE_PHRASES} stop={STOP_PHRASES} mute={TTS_MUTE_DURATION}s")
    log.info(f"PIPELINE Translator:    wake={TRANSLATE_WAKE!r} stops={TRANSLATE_STOPS}")
    log.info(f"PIPELINE Adult mode:    wake={ADULT_WAKE!r} stop={ADULT_STOP!r}")
    log.info(f"PIPELINE Athena variants: {ATHENA_WAKE_VARIANTS}")

    # ── v57: PortAudio device indexes (output and input). No PulseAudio. ──
    # find_pa_device walks sd.query_devices() looking for a name substring match
    # and the right channel direction. Same lookup the mic already uses.
    headset_pattern = LLM_CONFIG.get("headset_alsa_pattern", "Composite Device")
    external_pattern = LLM_CONFIG.get("external_alsa_pattern", "UACDemoV1")
    path_b_voice = LLM_CONFIG.get("path_b_voice", "am_michael")
    path_b_speed = float(LLM_CONFIG.get("path_b_speed", 0.85))

    headset_device = find_pa_device(headset_pattern, kind="output")
    external_device = find_pa_device(external_pattern, kind="output")
    headset_in_idx = find_pa_device(headset_pattern, kind="input")

    if headset_device is None:
        log.error(f"PIPELINE !! USB Composite Device output NOT FOUND in PortAudio "
                  f"(pattern={headset_pattern!r}) — wireless headset speaker will NOT play.")
    else:
        log.info(f"PIPELINE Wireless headset OUTPUT LOCKED: PA index {headset_device}")

    if headset_in_idx is None:
        log.error(f"PIPELINE !! USB Composite Device input NOT FOUND in PortAudio "
                  f"(pattern={headset_pattern!r}) — wireless headset mic will NOT capture.")
    else:
        log.info(f"PIPELINE Wireless headset MIC LOCKED:    PA index {headset_in_idx}")

    if external_device is None:
        log.error(f"PIPELINE !! UACDemoV1.0 output NOT FOUND in PortAudio "
                  f"(pattern={external_pattern!r}) — translation Path B will NOT play.")
    else:
        log.info(f"PIPELINE Wired USB speaker OUTPUT LOCKED: PA index {external_device} "
                 f"(voice={path_b_voice} speed={path_b_speed})")

    # v55 #20 / v60: load persisted Settings (schema v2). Voice, voice speed,
    # translation speed, and per-bucket translation voice gender all live here.
    runtime_state = load_runtime_state()
    runtime_state.setdefault("_version", 2)

    persisted_sink_label = runtime_state.get("active_output_sink_label", "headset")
    if persisted_sink_label == "speaker" and external_device is not None:
        active_output_device_initial = external_device
        log.info(f"PIPELINE Active output restored: speaker (PA index {external_device})")
    else:
        active_output_device_initial = headset_device
        if persisted_sink_label != "headset":
            log.info(f"PIPELINE Active output falling back to headset (persisted={persisted_sink_label!r})")
        else:
            log.info(f"PIPELINE Active output: headset (PA index {headset_device})")

    # v60: voice / speed / translation defaults from runtime state
    voice_options = LLM_CONFIG.get("voice_options", {"sky": "af_sky"})
    speed_grid = LLM_CONFIG.get("speed_grid", [0.68, 0.76, 0.84, 0.92, 1.00, 1.08, 1.16, 1.24, 1.32])

    persisted_voice_id = runtime_state.get("voice_name", TTS_VOICE)
    persisted_voice_speed_step = int(runtime_state.get("voice_speed_step", 5))
    persisted_translation_speed_step = int(runtime_state.get("translation_speed_step", 3))
    if not (1 <= persisted_voice_speed_step <= len(speed_grid)):
        persisted_voice_speed_step = 5
    if not (1 <= persisted_translation_speed_step <= len(speed_grid)):
        persisted_translation_speed_step = 3
    active_voice = persisted_voice_id
    active_voice_speed = speed_grid[persisted_voice_speed_step - 1]
    active_translation_speed = speed_grid[persisted_translation_speed_step - 1]

    # v60 (refined): single global translation voice gender — applied to all
    # language buckets uniformly. Default "male" per user spec; set to "female"
    # in a copy of the install for a different user.
    default_gender = LLM_CONFIG.get("translation_voice_gender_default", "male")
    persisted_gender = runtime_state.get("translation_voice_gender", default_gender)
    # Back-compat: if an old per-bucket dict shape sneaks through, take the
    # fallback bucket's value (or first key) and reduce to a single string.
    if isinstance(persisted_gender, dict):
        persisted_gender = (persisted_gender.get("fallback")
                            or next(iter(persisted_gender.values()), default_gender))
    if persisted_gender not in ("male", "female"):
        persisted_gender = default_gender
    translation_gender = persisted_gender

    log.info(f"PIPELINE v60 voice: {active_voice} @ step {persisted_voice_speed_step} ({active_voice_speed:.2f})")
    log.info(f"PIPELINE v60 translation speed: step {persisted_translation_speed_step} ({active_translation_speed:.2f})")
    log.info(f"PIPELINE v60 translation voice gender (global): {translation_gender}")

    # Wait for TTS + Whisper to come online (they start after LLM loads)
    tts_ok, whisper_ok = wait_for_services()

    # Startup greeting — first call processes system prompt, confirms Athena is online
    log.info("PIPELINE Startup greeting...")
    greeting = send_to_llm("Hello Athena", conversation, llm)
    if greeting:
        clean = clean_for_tts(greeting)
        if clean:
            tts_audio = send_to_tts(clean, voice=active_voice, speed=active_voice_speed) if tts_ok else None
            if tts_audio:
                play_to_device(tts_audio, active_output_device_initial)
                log.info(f"PIPELINE Athena greeted: '{clean}'")
            else:
                log.info(f"PIPELINE Athena (text): '{clean}'")
                print(f"\nAthena: {clean}\n")
    else:
        log.warning("PIPELINE No greeting from LLM")

    # v62: warm-idle memory snapshot — the moment Athena is fully booted and
    # idle, waiting for a wake-word. Captures the steady-state baseline that
    # earlier logs missed (host snapshots fired during cold start before the
    # LLM finished mmap'ing weight pages into Cached).
    memory_snapshot("warm-idle (post-greeting, listening)")

    is_speaking = False
    vad_muted = False
    silence_start = None
    speech_buffer = []
    pre_buffer = collections.deque(maxlen=PRE_BUFFER_CHUNKS)
    consecutive_speech = 0
    speech_confirmed = False
    mic_chunk_samples = int(VAD_CHUNK * (MIC_RATE / VAD_RATE))

    # ── Voice Router state ──
    is_idle = False                    # gate state; start active in conversation
    current_mode = "conversation"      # "conversation" | "adult" | "translate" | "settings" | None (idle)
    last_active_mode = "conversation"  # what to restore on wake from idle
    tts_muted = False
    tts_muted_until = 0.0
    translate_language = None
    translate_language_name = None
    translate_conversation = [{"role": "system", "content": "You are a translator."}]
    saved_translate_language = None
    saved_translate_language_name = None

    # ── Settings mode state (v54) ──
    settings_substate = None          # None | "awaiting_command" | "awaiting_output_value"
    settings_previous_mode = "conversation"

    # ── Active output sink (v54) ──
    # v55 #20: initialized from persisted Settings, computed above.
    # The Settings menu can switch this between headset_device and external_device at runtime.
    # Path B (translator foreign output) defaults to external_device (#16 makes that
    # configurable via path_b_force_external_device in llm_config.json).
    active_output_device = active_output_device_initial

    # v55 #21: track consecutive non-English Whisper detections for translate
    # auto-lock so a single misdetection can't lock the language for the session.
    translate_pending_lang = None
    translate_pending_count = 0
    translate_lock_threshold = int(LLM_CONFIG.get("translate_lock_consecutive", 2))
    path_b_force_external = bool(LLM_CONFIG.get("path_b_force_external_device", True))

    # Confidence monitor — logs peak VAD confidence every 3s while idle
    monitor_peak = 0.0
    monitor_time = time.time()

    log.info(f"PIPELINE Mic chunk: {mic_chunk_samples} samples = {mic_chunk_samples/MIC_RATE*1000:.0f}ms")
    log.info("PIPELINE Listening... (speak to activate)")

    # Lock mic input to the USB Composite Device's INPUT (looked up independently
    # of the output side, so a muted/disabled speaker can't break the mic and
    # vice-versa). If not found, fall back to PortAudio default and log it.
    if headset_in_idx is not None:
        log.info(f"PIPELINE Mic input LOCKED: PA index {headset_in_idx}")
    else:
        log.error("PIPELINE Mic input NOT locked — using PortAudio default")

    # v56: reverted to v54 blocking-read form. The v55 callback-driven open
    # broke the device inside the container (PortAudio's ALSA callback path
    # returned paDeviceUnavailable). The "Mic overflow" warning that v55
    # tried to suppress was cosmetic — keep it, mic stays working.
    try:
        with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="float32",
                            blocksize=mic_chunk_samples, device=headset_in_idx) as stream:
            while True:
                audio_chunk, overflowed = stream.read(mic_chunk_samples)
                if overflowed: log.warning("PIPELINE Mic overflow")

                if vad_muted:
                    continue

                audio_mic = audio_chunk.flatten()
                audio_16k = resample_audio(audio_mic, MIC_RATE, VAD_RATE)
                vad_input = audio_16k[:VAD_CHUNK] if len(audio_16k) >= VAD_CHUNK else np.pad(audio_16k, (0, VAD_CHUNK - len(audio_16k)))

                confidence = vad.process_chunk(vad_input)

                if not is_speaking:
                    pre_buffer.append(audio_mic.copy())

                if not is_speaking:
                    if confidence > monitor_peak:
                        monitor_peak = confidence
                    if time.time() - monitor_time >= 3.0:
                        log.info(f"PIPELINE [monitor] peak_conf={monitor_peak:.3f}  thresh={VAD_THRESHOLD}")
                        monitor_peak = 0.0
                        monitor_time = time.time()

                if confidence >= VAD_THRESHOLD:
                    consecutive_speech += 1
                    if not speech_confirmed and consecutive_speech >= MIN_SPEECH_FRAMES:
                        speech_confirmed = True
                        is_speaking = True
                        speech_buffer = list(pre_buffer)
                        pre_buffer.clear()
                        log.info(f"PIPELINE >> Speech CONFIRMED (conf={confidence:.2f}, {consecutive_speech} frames, pre-buf={len(speech_buffer)} chunks)")
                    if is_speaking:
                        speech_buffer.append(audio_mic.copy())
                        silence_start = None
                else:
                    consecutive_speech = 0
                    if is_speaking:
                        speech_buffer.append(audio_mic.copy())
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start >= SILENCE_TIMEOUT:
                            duration = len(speech_buffer) * mic_chunk_samples / MIC_RATE
                            log.info(f"PIPELINE << Speech END ({duration:.1f}s)")
                            if duration >= MIN_SPEECH_DURATION:
                                full = np.concatenate(speech_buffer)
                                wav = audio_to_wav_bytes(resample_audio(full, MIC_RATE, WHISPER_RATE), WHISPER_RATE)
                                vad_muted = True
                                log.info("PIPELINE VAD muted (processing)")

                                # v55 #17: pass recording duration so timeout scales appropriately.
                                # v55 #26: conversation/adult uses default auto-detect — non-English
                                # transcripts get dropped below.
                                if current_mode == "translate":
                                    text, detected_lang = send_to_whisper(wav, language_override="", translate_override=True, duration_hint=duration)
                                else:
                                    text, detected_lang = send_to_whisper(wav, duration_hint=duration)

                                if text and text not in ("", "[BLANK_AUDIO]", "(blank audio)"):
                                    log.info(f"PIPELINE Transcript: '{text}' (lang={detected_lang!r})")

                                    # v55 #26: Athena is English-only outside translate mode.
                                    # If Whisper detected non-English in conversation/adult,
                                    # drop the transcript silently so accidental foreign speech
                                    # doesn't get sent to the LLM as garbled English.
                                    if (current_mode in ("conversation", "adult")
                                            and detected_lang
                                            and detected_lang.lower() not in ("english", "en", "")):
                                        log.info(f"PIPELINE Non-English ({detected_lang}) in {current_mode} — dropped")
                                        vad_muted = False
                                        is_speaking = False
                                        speech_confirmed = False
                                        silence_start = None
                                        speech_buffer = []
                                        pre_buffer.clear()
                                        consecutive_speech = 0
                                        vad.reset()
                                        monitor_peak = 0.0
                                        monitor_time = time.time()
                                        log.info("PIPELINE Listening...")
                                        continue

                                    text_lower = text.strip().lower().replace(",", "")

                                    # TTS mute timeout check
                                    if tts_muted and time.time() >= tts_muted_until:
                                        tts_muted = False
                                        log.info("PIPELINE TTS unmuted (timeout expired)")

                                    gate_pass = False
                                    skip_llm_direct_tts = False
                                    forwarded_text = text  # what we actually send to the LLM (post-strip)

                                    def _strip_phrase(orig_text, phrase):
                                        """Remove phrase (and its athena variants) from orig_text
                                        case-insensitively, tidy result. v61: when phrase contains
                                        'athena', try all variants so a misheard wake-word still
                                        gets stripped from the remainder forwarded to the LLM."""
                                        candidates = [phrase]
                                        if "athena" in phrase.lower():
                                            candidates = [phrase.lower().replace("athena", v)
                                                          for v in ATHENA_WAKE_VARIANTS]
                                        cleaned = orig_text
                                        for cand in candidates:
                                            cleaned = re.sub(re.escape(cand), "", cleaned, flags=re.IGNORECASE)
                                        cleaned = re.sub(r'\s+', ' ', cleaned).strip(" ,.;:!?—–-")
                                        return cleaned

                                    # Pre-compute stop matches (universal "athena stop" is a substring
                                    # of "athena stop translating" — disambiguate explicitly).
                                    # v61: athena_match handles wake-word variants automatically.
                                    stop_match = any(athena_match(text_lower, p) for p in STOP_PHRASES)
                                    translate_stop_match = any(athena_match(text_lower, s) for s in TRANSLATE_STOPS)

                                    # ─── ROUTER ───
                                    # 0. Translate-specific stop (priority over universal stop because
                                    #    "athena stop translating" CONTAINS "athena stop"). Silent.
                                    if current_mode == "translate" and translate_stop_match:
                                        kill_active_paplay()  # v55 #2
                                        tts_muted = True
                                        tts_muted_until = time.time() + TTS_MUTE_DURATION
                                        current_mode = "conversation"
                                        translate_language = None
                                        translate_language_name = None
                                        translate_pending_lang = None
                                        translate_pending_count = 0
                                        log.info(f"PIPELINE → CONVERSATION (translate ended) — TTS muted {TTS_MUTE_DURATION}s")

                                    # 1. Universal STOP — any active mode → IDLE. Always silent.
                                    # v55 #11: also fires when "athena stop translating" is said while
                                    # NOT in translate mode (was previously falling through to LLM).
                                    elif (not is_idle and stop_match
                                          and (not translate_stop_match or current_mode != "translate")):
                                        kill_active_paplay()  # v55 #2
                                        tts_muted = True
                                        tts_muted_until = time.time() + TTS_MUTE_DURATION
                                        # If we were inside settings, treat the previous mode as the saved one
                                        if current_mode == "settings":
                                            last_active_mode = settings_previous_mode
                                        else:
                                            last_active_mode = current_mode
                                        if last_active_mode == "translate":
                                            saved_translate_language = translate_language
                                            saved_translate_language_name = translate_language_name
                                        is_idle = True
                                        current_mode = None
                                        settings_substate = None
                                        log.info(f"PIPELINE → IDLE (saved={last_active_mode}) — TTS muted {TTS_MUTE_DURATION}s")

                                    # 2. Wake from idle — restore last active mode. STRIP wake phrase
                                    #    from text; if remainder, send remainder to LLM.
                                    # v55 #9: adult-mode restore swaps ONLY the system prompt at index 0,
                                    # preserving conversation history across the idle round-trip.
                                    # v55 #10: after stripping the wake phrase, re-check the remainder
                                    # against mode-switch phrases (e.g., "okay athena adult mode" should
                                    # actually enter ADULT, not forward "adult mode" to the LLM).
                                    elif is_idle and athena_any_match(text_lower, WAKE_PHRASES):
                                        is_idle = False
                                        current_mode = last_active_mode
                                        for p in WAKE_PHRASES:
                                            if athena_match(text_lower, p):
                                                forwarded_text = _strip_phrase(text, p)
                                                break
                                        if current_mode == "translate":
                                            translate_language = saved_translate_language
                                            translate_language_name = saved_translate_language_name
                                            if translate_language_name:
                                                translate_conversation = [{
                                                    "role": "system",
                                                    "content": LLM_CONFIG.get("translate_prompt", "Translate to {language}").format(language=translate_language_name)
                                                }]
                                            log.info(f"PIPELINE → TRANSLATE MODE (resumed, lang={translate_language_name})")
                                        elif current_mode == "adult":
                                            if conversation and conversation[0].get("role") == "system":
                                                conversation[0] = {"role": "system", "content": adult_system}
                                            else:
                                                conversation = [{"role": "system", "content": adult_system}] + conversation
                                            log.info("PIPELINE → ADULT MODE (resumed, history preserved)")
                                        else:
                                            log.info("PIPELINE → CONVERSATION MODE (resumed)")

                                        # v55 #10: re-check the remainder for mode-switch phrases.
                                        rem_lower = (forwarded_text or "").lower()
                                        if rem_lower:
                                            if athena_match(rem_lower, SETTINGS_WAKE) and current_mode in ("conversation", "adult"):
                                                kill_active_paplay()
                                                tts_muted = False
                                                tts_muted_until = 0.0
                                                settings_previous_mode = current_mode
                                                current_mode = "settings"
                                                settings_substate = "awaiting_command"
                                                forwarded_text = ""
                                                log.info(f"PIPELINE   wake-then-Settings (from {settings_previous_mode})")
                                                if active_output_device is not None:
                                                    prompt_wav = send_to_tts("Settings. Say audio output.")
                                                    if prompt_wav:
                                                        play_to_device(prompt_wav, active_output_device)
                                            elif athena_match(rem_lower, ADULT_WAKE) and current_mode == "conversation":
                                                # v59: wipe history on mode switch (even via wake)
                                                conversation = [{"role": "system", "content": adult_system}]
                                                current_mode = "adult"
                                                forwarded_text = _strip_phrase(forwarded_text, ADULT_WAKE)
                                                log.info(f"PIPELINE   wake-then-ADULT (history WIPED, remainder='{forwarded_text[:60]}')")
                                                if forwarded_text:
                                                    gate_pass = True
                                            elif athena_match(rem_lower, ADULT_STOP) and current_mode == "adult":
                                                # v59: wipe history on mode switch (even via wake)
                                                conversation = [{"role": "system", "content": normal_system}]
                                                current_mode = "conversation"
                                                forwarded_text = _strip_phrase(forwarded_text, ADULT_STOP)
                                                log.info(f"PIPELINE   wake-then-CONVERSATION (history WIPED, remainder='{forwarded_text[:60]}')")
                                                if forwarded_text:
                                                    gate_pass = True
                                            elif athena_match(rem_lower, TRANSLATE_WAKE) and current_mode == "conversation":
                                                kill_active_paplay()
                                                current_mode = "translate"
                                                translate_language = None
                                                translate_language_name = None
                                                translate_conversation = [{"role": "system", "content": "You are a translator."}]
                                                forwarded_text = ""
                                                log.info("PIPELINE   wake-then-TRANSLATE (say a language)")
                                            else:
                                                gate_pass = True
                                                log.info(f"PIPELINE   resumed with remainder: '{forwarded_text[:80]}'")
                                        else:
                                            log.info("PIPELINE   silent resume (no remainder)")

                                    # 3. Idle and not wake — drop transcript silently
                                    elif is_idle:
                                        log.info(f"PIPELINE IDLE — discarded: '{text[:60]}'")

                                    # 4. Adult mode entry (from conversation OR thinking). STRIP phrase.
                                    # v62: PRESERVE history. The "one-way trick" — adult inherits
                                    # everything you've been talking about so it can add flair to it.
                                    # The destructive cleanup happens only on the adult → conversation
                                    # exit (branch 5). System-prompt swap in place.
                                    # 4. Adult mode entry (from conversation only). STRIP phrase.
                                    # v59: WIPE conversation history on mode change. Idle-resume keeps
                                    # history; explicit mode switches start fresh per user request.
                                    elif athena_match(text_lower, ADULT_WAKE) and current_mode == "conversation":
                                        current_mode = "adult"
                                        conversation = [{"role": "system", "content": adult_system}]
                                        forwarded_text = _strip_phrase(text, ADULT_WAKE)
                                        log.info(f"PIPELINE → ADULT MODE (thinking={'ON' if adult_thinking else 'OFF'}, history WIPED)")
                                        if forwarded_text:
                                            gate_pass = True
                                            log.info(f"PIPELINE   adult-entry remainder: '{forwarded_text[:80]}'")
                                        else:
                                            log.info("PIPELINE   silent adult-entry (no remainder)")

                                    # 5. Adult-specific stop (from adult). STRIP phrase.
                                    # v59: WIPE conversation history on mode change.
                                    elif athena_match(text_lower, ADULT_STOP) and current_mode == "adult":
                                        current_mode = "conversation"
                                        conversation = [{"role": "system", "content": normal_system}]
                                        forwarded_text = _strip_phrase(text, ADULT_STOP)
                                        log.info("PIPELINE → CONVERSATION (from adult, history WIPED)")
                                        if forwarded_text:
                                            gate_pass = True
                                            log.info(f"PIPELINE   normal-resume remainder: '{forwarded_text[:80]}'")
                                        else:
                                            log.info("PIPELINE   silent return to conversation")

                                    # 6. Translate entry (from conversation only). Silent (waiting for language).
                                    elif athena_match(text_lower, TRANSLATE_WAKE) and current_mode == "conversation":
                                        kill_active_paplay()  # v55 #2
                                        current_mode = "translate"
                                        translate_language = None
                                        translate_language_name = None
                                        translate_pending_lang = None
                                        translate_pending_count = 0
                                        translate_conversation = [{"role": "system", "content": "You are a translator."}]
                                        log.info("PIPELINE → TRANSLATE MODE — say a language")

                                    # 6.5. Settings entry (v60: expanded sub-menu). Force-clear tts_muted
                                    # so prompts always play. Settings is a STOP-equivalent action.
                                    elif athena_match(text_lower, SETTINGS_WAKE) and current_mode in ("conversation", "adult"):
                                        kill_active_paplay()
                                        tts_muted = False
                                        tts_muted_until = 0.0
                                        settings_previous_mode = current_mode
                                        current_mode = "settings"
                                        settings_substate = "awaiting_command"
                                        log.info(f"PIPELINE → SETTINGS (from {settings_previous_mode})")
                                        if active_output_device is not None:
                                            menu_msg = ("Settings. Say audio output, voice, voice speed, "
                                                        "translation speed, or translation voice. "
                                                        "Say exit settings when done.")
                                            prompt_wav = send_to_tts(menu_msg, voice=active_voice, speed=active_voice_speed)
                                            if prompt_wav:
                                                play_to_device(prompt_wav, active_output_device)

                                    # 6.6. In SETTINGS mode (v60 refined). After each change the menu
                                    # stays open at awaiting_command — user explicitly says "exit
                                    # settings" / "done" to leave. Single global "translation voice"
                                    # M/F applies to all language buckets.
                                    elif current_mode == "settings":
                                        # Universal exit (no change persisted)
                                        if athena_any_match(text_lower, SETTINGS_EXIT_PHRASES):
                                            current_mode = settings_previous_mode
                                            settings_substate = None
                                            log.info(f"PIPELINE → {settings_previous_mode.upper()} (settings exit)")

                                        elif settings_substate == "awaiting_command":
                                            # v61 #3: re-play the menu prompt without changing state.
                                            if any(w in text_lower for w in ("options", "help", "list", "menu")):
                                                log.info("PIPELINE SETTINGS: re-prompting menu (help/options/list/menu)")
                                                if active_output_device is not None:
                                                    menu_msg = ("Settings. Say audio output, voice, voice speed, "
                                                                "translation speed, or translation voice. "
                                                                "Say exit settings when done.")
                                                    prompt_wav = send_to_tts(menu_msg, voice=active_voice, speed=active_voice_speed)
                                                    if prompt_wav:
                                                        play_to_device(prompt_wav, active_output_device)
                                            elif "translation voice" in text_lower:
                                                settings_substate = "awaiting_translation_voice_value"
                                                log.info("PIPELINE SETTINGS: translation voice — awaiting male/female")
                                                if active_output_device is not None:
                                                    prompt_wav = send_to_tts("Male or female?",
                                                                             voice=active_voice, speed=active_voice_speed)
                                                    if prompt_wav:
                                                        play_to_device(prompt_wav, active_output_device)

                                            elif "voice speed" in text_lower:
                                                settings_substate = "awaiting_voice_speed_value"
                                                log.info("PIPELINE SETTINGS: voice speed — awaiting 1-9")
                                                if active_output_device is not None:
                                                    prompt_wav = send_to_tts("Say a number from one to nine. Five is normal.",
                                                                             voice=active_voice, speed=active_voice_speed)
                                                    if prompt_wav:
                                                        play_to_device(prompt_wav, active_output_device)

                                            elif "translation speed" in text_lower:
                                                settings_substate = "awaiting_translation_speed_value"
                                                log.info("PIPELINE SETTINGS: translation speed — awaiting 1-9")
                                                if active_output_device is not None:
                                                    prompt_wav = send_to_tts("Say a number from one to nine.",
                                                                             voice=active_voice, speed=active_voice_speed)
                                                    if prompt_wav:
                                                        play_to_device(prompt_wav, active_output_device)

                                            elif "voice" in text_lower:
                                                settings_substate = "awaiting_voice_value"
                                                names = ", ".join(voice_options.keys())
                                                log.info(f"PIPELINE SETTINGS: voice — awaiting one of {names}")
                                                if active_output_device is not None:
                                                    prompt_wav = send_to_tts(f"Say one of: {names}.",
                                                                             voice=active_voice, speed=active_voice_speed)
                                                    if prompt_wav:
                                                        play_to_device(prompt_wav, active_output_device)

                                            elif SETTINGS_AUDIO_PHRASE in text_lower:
                                                settings_substate = "awaiting_output_value"
                                                log.info("PIPELINE SETTINGS: audio output — awaiting headphones/speaker")
                                                if active_output_device is not None:
                                                    prompt_wav = send_to_tts("Headphones or speaker?",
                                                                             voice=active_voice, speed=active_voice_speed)
                                                    if prompt_wav:
                                                        play_to_device(prompt_wav, active_output_device)
                                            else:
                                                log.info(f"PIPELINE SETTINGS: unrecognized command '{text[:60]}'")

                                        elif settings_substate == "awaiting_output_value":
                                            new_dev = None
                                            confirm_text = ""
                                            new_label = None
                                            if SETTINGS_SPEAKER in text_lower:
                                                new_dev = external_device
                                                confirm_text = "Audio output set to speaker."
                                                new_label = "speaker"
                                            elif SETTINGS_HEADPHONES in text_lower:
                                                new_dev = headset_device
                                                confirm_text = "Audio output set to headphones."
                                                new_label = "headphones"
                                            if new_dev is not None:
                                                active_output_device = new_dev
                                                log.info(f"PIPELINE SETTINGS: active_output_device → PA index {active_output_device}")
                                                runtime_state["active_output_sink_label"] = (
                                                    "speaker" if new_label == "speaker" else "headset"
                                                )
                                                save_runtime_state(runtime_state)
                                                confirm_wav = send_to_tts(confirm_text, voice=active_voice, speed=active_voice_speed)
                                                if confirm_wav:
                                                    play_to_device(confirm_wav, active_output_device)
                                                # v60 refined: stay in settings; user says "exit settings" to leave.
                                                settings_substate = "awaiting_command"
                                                log.info("PIPELINE SETTINGS: change applied, awaiting next command (say exit settings to leave)")
                                            else:
                                                log.info(f"PIPELINE SETTINGS: expected speaker/headphones, got '{text[:60]}'")

                                        elif settings_substate == "awaiting_voice_value":
                                            picked = None
                                            for nick, vid in voice_options.items():
                                                if nick in text_lower:
                                                    picked = (nick, vid); break
                                            if picked:
                                                active_voice = picked[1]
                                                runtime_state["voice_name"] = active_voice
                                                save_runtime_state(runtime_state)
                                                log.info(f"PIPELINE SETTINGS: voice → {picked[0]} ({active_voice})")
                                                confirm_wav = send_to_tts(f"Voice set to {picked[0]}.", voice=active_voice, speed=active_voice_speed)
                                                if confirm_wav and active_output_device is not None:
                                                    play_to_device(confirm_wav, active_output_device)
                                                settings_substate = "awaiting_command"
                                                log.info("PIPELINE SETTINGS: change applied, awaiting next command (say exit settings to leave)")
                                            else:
                                                log.info(f"PIPELINE SETTINGS: voice not recognized in '{text[:60]}'")

                                        elif settings_substate == "awaiting_voice_speed_value":
                                            step = _parse_speed_step(text_lower)
                                            if step and 1 <= step <= len(speed_grid):
                                                active_voice_speed = speed_grid[step - 1]
                                                runtime_state["voice_speed_step"] = step
                                                save_runtime_state(runtime_state)
                                                log.info(f"PIPELINE SETTINGS: voice speed → step {step} ({active_voice_speed:.2f})")
                                                confirm_wav = send_to_tts(f"Voice speed set to {step}.", voice=active_voice, speed=active_voice_speed)
                                                if confirm_wav and active_output_device is not None:
                                                    play_to_device(confirm_wav, active_output_device)
                                                settings_substate = "awaiting_command"
                                                log.info("PIPELINE SETTINGS: change applied, awaiting next command (say exit settings to leave)")
                                            else:
                                                log.info(f"PIPELINE SETTINGS: voice speed value not recognized in '{text[:60]}'")

                                        elif settings_substate == "awaiting_translation_speed_value":
                                            step = _parse_speed_step(text_lower)
                                            if step and 1 <= step <= len(speed_grid):
                                                active_translation_speed = speed_grid[step - 1]
                                                runtime_state["translation_speed_step"] = step
                                                save_runtime_state(runtime_state)
                                                log.info(f"PIPELINE SETTINGS: translation speed → step {step} ({active_translation_speed:.2f})")
                                                confirm_wav = send_to_tts(f"Translation speed set to {step}.",
                                                                          voice=active_voice, speed=active_voice_speed)
                                                if confirm_wav and active_output_device is not None:
                                                    play_to_device(confirm_wav, active_output_device)
                                                settings_substate = "awaiting_command"
                                                log.info("PIPELINE SETTINGS: change applied, awaiting next command (say exit settings to leave)")
                                            else:
                                                log.info(f"PIPELINE SETTINGS: translation speed not recognized in '{text[:60]}'")

                                        elif settings_substate == "awaiting_translation_voice_value":
                                            new_gender = None
                                            if "female" in text_lower:
                                                new_gender = "female"
                                            elif "male" in text_lower:
                                                new_gender = "male"
                                            if new_gender:
                                                translation_gender = new_gender
                                                runtime_state["translation_voice_gender"] = new_gender
                                                save_runtime_state(runtime_state)
                                                log.info(f"PIPELINE SETTINGS: translation_voice_gender → {new_gender} (global)")
                                                confirm_wav = send_to_tts(f"Translation voice set to {new_gender}.",
                                                                          voice=active_voice, speed=active_voice_speed)
                                                if confirm_wav and active_output_device is not None:
                                                    play_to_device(confirm_wav, active_output_device)
                                                settings_substate = "awaiting_command"
                                                log.info("PIPELINE SETTINGS: change applied, awaiting next command (say exit settings to leave)")
                                            else:
                                                log.info(f"PIPELINE SETTINGS: expected male/female, got '{text[:60]}'")

                                    # 7. In translate mode — language locking + paths A/B
                                    # v55 #21: "athena unlock language" clears an auto-locked language
                                    # without leaving translate mode. Auto-lock from Whisper now
                                    # requires N consecutive non-English detections to avoid locking
                                    # on a single misdetection.
                                    elif current_mode == "translate":
                                        # Unlock command (v55 #21)
                                        if athena_any_match(text_lower, TRANSLATE_UNLOCK_PHRASES):
                                            translate_language = None
                                            translate_language_name = None
                                            translate_pending_lang = None
                                            translate_pending_count = 0
                                            translate_conversation = [{"role": "system", "content": "You are a translator."}]
                                            log.info("PIPELINE TRANSLATE: language UNLOCKED (waiting for new language)")
                                            # No further action this turn — fall through to the
                                            # "no language locked" branch which logs and waits.

                                        speak_switch = False
                                        if athena_match(text_lower, "athena speak"):
                                            for lang_name in LANGUAGE_MAP:
                                                if lang_name in text_lower:
                                                    translate_language = lang_name
                                                    translate_language_name = lang_name.capitalize()
                                                    translate_conversation = [{
                                                        "role": "system",
                                                        "content": LLM_CONFIG.get("translate_prompt", "Translate to {language}").format(language=translate_language_name)
                                                    }]
                                                    translate_pending_lang = None
                                                    translate_pending_count = 0
                                                    speak_switch = True
                                                    log.info(f"PIPELINE TRANSLATE: switched to {translate_language_name}")
                                                    break
                                        language_just_locked = False
                                        if not speak_switch and not translate_language:
                                            for lang_name in LANGUAGE_MAP:
                                                if lang_name in text_lower:
                                                    translate_language = lang_name
                                                    translate_language_name = lang_name.capitalize()
                                                    translate_conversation = [{
                                                        "role": "system",
                                                        "content": LLM_CONFIG.get("translate_prompt", "Translate to {language}").format(language=translate_language_name)
                                                    }]
                                                    translate_pending_lang = None
                                                    translate_pending_count = 0
                                                    language_just_locked = True
                                                    log.info(f"PIPELINE TRANSLATE: language locked to {translate_language_name}")
                                                    break
                                        # v55 #21: N-consecutive auto-lock from Whisper detection.
                                        if (not speak_switch and not translate_language
                                                and detected_lang
                                                and detected_lang.lower() not in ("english", "en", "")):
                                            dl = detected_lang.lower()
                                            if dl == translate_pending_lang:
                                                translate_pending_count += 1
                                            else:
                                                translate_pending_lang = dl
                                                translate_pending_count = 1
                                            log.info(f"PIPELINE TRANSLATE: auto-detect {dl} ({translate_pending_count}/{translate_lock_threshold})")
                                            if translate_pending_count >= translate_lock_threshold:
                                                translate_language = dl
                                                translate_language_name = detected_lang.capitalize()
                                                translate_conversation = [{
                                                    "role": "system",
                                                    "content": LLM_CONFIG.get("translate_prompt", "Translate to {language}").format(language=translate_language_name)
                                                }]
                                                translate_pending_lang = None
                                                translate_pending_count = 0
                                                language_just_locked = True
                                                log.info(f"PIPELINE TRANSLATE: auto-locked to {translate_language_name}")

                                        if speak_switch or language_just_locked:
                                            log.info(f"PIPELINE TRANSLATE: language set to {translate_language_name} — ready")
                                        elif detected_lang and detected_lang.lower() not in ("english", "en", ""):
                                            # Path A: foreign → English (Whisper translated server-side)
                                            skip_llm_direct_tts = True
                                            gate_pass = True
                                            log.info(f"PIPELINE TRANSLATE PATH A: foreign→English '{text[:60]}'")
                                        elif translate_language:
                                            # Path B: English → foreign (LLM renders translation)
                                            gate_pass = True
                                            log.info(f"PIPELINE TRANSLATE PATH B: English→{translate_language_name} '{text[:60]}'")
                                        else:
                                            log.info("PIPELINE TRANSLATE: no language locked — say a language name")

                                    # 8. Active conversation/adult — pass to LLM
                                    else:
                                        gate_pass = True

                                    # v64: per-turn THINK trigger applied just before each LLM call
                                    # (see streaming and non-streaming dispatch branches below).

                                    # ─── PROCESS GATED SPEECH ───
                                    if gate_pass:
                                        is_path_b = (current_mode == "translate"
                                                     and translate_language
                                                     and not skip_llm_direct_tts)
                                        # v55 #8: when both sinks are None, this MUST be False.
                                        external_speaker_active = (
                                            external_device is not None
                                            and active_output_device is not None
                                            and active_output_device == external_device
                                        )

                                        # v60 dual-channel translate routing:
                                        #   Path A (foreign → English) → headset_device, always.
                                        #   Path B (English → foreign) → external_device, always.
                                        # In conversation/adult mode, audio uses active_output_device
                                        # per the user's Settings choice. Translate mode is fixed.
                                        path_a_device = headset_device if headset_device is not None else active_output_device
                                        if external_device is not None:
                                            path_b_device = external_device
                                        else:
                                            path_b_device = active_output_device
                                            if is_path_b:
                                                log.warning("PIPELINE Path B: external_device unavailable "
                                                            "— falling back to active_output_device")

                                        if skip_llm_direct_tts:
                                            # Path A: Whisper already translated to English. Always plays
                                            # on the headset for the dual-channel use case (English speaker
                                            # listens to translation on headset, foreign speaker hears
                                            # path B on the speaker).
                                            clean = text
                                            if clean and not tts_muted:
                                                tts_audio = send_to_tts(clean, voice=active_voice, speed=active_voice_speed, lang="en-us")
                                                if tts_audio:
                                                    log.info(f"PIPELINE Path A (foreign→EN): device={path_a_device}")
                                                    play_to_device(tts_audio, path_a_device)
                                                else:
                                                    print(f"\nAthena: {clean}\n")
                                            translation_log.info(f"[{time.strftime('%H:%M:%S')}] PATH A {detected_lang or '?'}→EN  src='{text}'")
                                            vad_muted = False
                                            log.info("PIPELINE VAD unmuted (Path A done)")

                                        elif is_path_b:
                                            # Path B: translate English → target language. Always plays
                                            # on external_device. Voice + lang resolved from the
                                            # path_b_voice_table using the user's bucket gender choice.
                                            tgt_lang_code, tgt_voice, tgt_bucket = resolve_path_b_voice(
                                                translate_language, translation_gender)
                                            log.info(f"PIPELINE Path B routing: target={translate_language} bucket={tgt_bucket} "
                                                     f"lang={tgt_lang_code} voice={tgt_voice}")

                                            reply = send_to_llm(forwarded_text or text, translate_conversation, llm)

                                            # Non-Latin double-pass. Triggered by bucket. The
                                            # path_b_voice_table carries lang="en-us" and the actual voices
                                            # (am_michael / af_heart) for these buckets so Kokoro/espeak only
                                            # ever sees English-letter pass-2 output spoken by an American voice.
                                            if tgt_bucket in ("chinese", "japanese", "hindi") and reply:
                                                asian_conv = [
                                                    {"role": "system", "content": ASIAN_OVERRIDE_PROMPT},
                                                    {"role": "user",   "content": reply},
                                                ]
                                                reply = send_to_llm(reply, asian_conv, llm)

                                            clean = clean_for_tts(reply) if reply else ""

                                            if clean and not tts_muted:
                                                trans_audio = send_to_tts(clean,
                                                                          voice=tgt_voice,
                                                                          speed=active_translation_speed,
                                                                          lang=tgt_lang_code)
                                                if trans_audio:
                                                    log.info(f"PIPELINE Path B: translation on device={path_b_device} "
                                                             f"voice={tgt_voice} lang={tgt_lang_code} "
                                                             f"speed={active_translation_speed:.2f}")
                                                    play_to_device(trans_audio, path_b_device)
                                                else:
                                                    print(f"\nAthena ({translate_language_name}): {clean}\n")
                                            elif not clean:
                                                log.warning("PIPELINE Empty translation reply")
                                            translation_log.info(f"[{time.strftime('%H:%M:%S')}] PATH B EN→{translate_language_name} "
                                                                 f"voice={tgt_voice} lang={tgt_lang_code}\n  EN:  {forwarded_text or text}\n  OUT: {clean}")
                                            vad_muted = False
                                            log.info("PIPELINE VAD unmuted (Path B done)")

                                        elif streaming_enabled and current_mode in ("conversation", "adult"):
                                            # v55 #4: streaming branch now respects tts_muted —
                                            # discard the prompt rather than play during the mute window.

                                            # v64: per-turn thinking trigger. If the user utterance
                                            # contains the whole word "think", swap conversation[0] to
                                            # the <|think|>-prefixed system prompt for THIS turn. Next
                                            # turn (no "think") swaps back. Only conversation/adult.
                                            base_prompt = LLM_CONFIG["adult_prompt"] if current_mode == "adult" else LLM_CONFIG["system_prompt"]
                                            if re.search(r'\bthink\b', text_lower):
                                                conversation[0] = {"role": "system", "content": "<|think|>" + base_prompt}
                                            else:
                                                conversation[0] = {"role": "system", "content": base_prompt}

                                            if tts_muted:
                                                log.info("PIPELINE TTS muted — discarding streaming reply")
                                                vad_muted = False
                                            elif external_speaker_active:
                                                log.info(f"PIPELINE Stream → external speaker {active_output_device}, "
                                                         f"mic dead through full stream")
                                                stream_llm_to_tts(forwarded_text, conversation, llm,
                                                                  device_idx=active_output_device,
                                                                  voice=active_voice, speed=active_voice_speed, lang="en-us",
                                                                  on_first_chunk=None)
                                                vad_muted = False
                                                log.info("PIPELINE VAD unmuted (stream done, external speaker)")
                                            else:
                                                mute_state = {"muted": True}
                                                def _on_first_chunk():
                                                    nonlocal vad_muted
                                                    vad_muted = False
                                                    mute_state["muted"] = False
                                                    log.info("PIPELINE VAD unmuted (first sentence queued)")
                                                stream_llm_to_tts(forwarded_text, conversation, llm,
                                                                  device_idx=active_output_device,
                                                                  voice=active_voice, speed=active_voice_speed, lang="en-us",
                                                                  on_first_chunk=_on_first_chunk)
                                                if mute_state["muted"]:
                                                    vad_muted = False
                                                    log.info("PIPELINE VAD unmuted (stream done, no chunks)")

                                        else:
                                            # Non-streaming conversation/adult fallback → active sink.
                                            # v64: per-turn thinking trigger (same logic as streaming).
                                            base_prompt = LLM_CONFIG["adult_prompt"] if current_mode == "adult" else LLM_CONFIG["system_prompt"]
                                            if re.search(r'\bthink\b', text_lower):
                                                conversation[0] = {"role": "system", "content": "<|think|>" + base_prompt}
                                            else:
                                                conversation[0] = {"role": "system", "content": base_prompt}
                                            reply = send_to_llm(forwarded_text, conversation, llm)
                                            clean = clean_for_tts(reply) if reply else ""
                                            if clean and not tts_muted:
                                                tts_audio = send_to_tts(clean, voice=active_voice, speed=active_voice_speed, lang="en-us")
                                                if tts_audio:
                                                    play_to_device(tts_audio, active_output_device)
                                                else:
                                                    print(f"\nAthena: {clean}\n")
                                            elif clean and tts_muted:
                                                log.info(f"PIPELINE TTS muted — discarding: '{clean[:60]}'")
                                            elif not clean:
                                                log.warning("PIPELINE Empty reply")
                                            vad_muted = False
                                            log.info("PIPELINE VAD unmuted (TTS done)")
                                    else:
                                        vad_muted = False
                                        log.info("PIPELINE VAD unmuted (no LLM call)")
                                else:
                                    vad_muted = False
                                    log.info(f"PIPELINE VAD unmuted (noise)")
                                    log.info(f"PIPELINE Noise: '{text}'")
                            else:
                                log.info(f"PIPELINE Too short ({duration:.1f}s)")
                            # v62: per-turn memory snapshot. Throttled so a fast
                            # back-and-forth doesn't fill the log; spaced at 15s
                            # we still get a clean picture of memory drift over
                            # a session without drowning it.
                            memory_snapshot(f"post-turn (mode={current_mode}, idle={is_idle})",
                                            throttle_seconds=15)
                            is_speaking = False
                            speech_confirmed = False
                            silence_start = None
                            speech_buffer = []
                            pre_buffer.clear()
                            consecutive_speech = 0
                            vad.reset()
                            monitor_peak = 0.0
                            monitor_time = time.time()
                            log.info("PIPELINE Listening...")
    except KeyboardInterrupt: log.info("PIPELINE Shutdown")
    except Exception as e:
        log.error(f"PIPELINE FATAL: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
