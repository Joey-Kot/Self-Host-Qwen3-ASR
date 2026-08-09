from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

from itn_numeric import apply_numeric_itn

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_NAME = os.getenv("ASR_MODEL_ID", "Qwen/Qwen3-ASR-1.7B").strip() or "Qwen/Qwen3-ASR-1.7B"
LOCAL_MODEL_DIR = Path(
    os.getenv("ASR_LOCAL_MODEL_DIR", str(PROJECT_DIR / "models" / "Qwen3-ASR-1.7B"))
).expanduser()
EXPLICIT_MODEL = os.getenv("ASR_MODEL", "").strip()

ALIGNER_MODEL_NAME = (
    os.getenv("ASR_ALIGNER_ID", "Qwen/Qwen3-ForcedAligner-0.6B").strip()
    or "Qwen/Qwen3-ForcedAligner-0.6B"
)
LOCAL_ALIGNER_DIR = Path(
    os.getenv("ASR_LOCAL_ALIGNER_DIR", str(PROJECT_DIR / "models" / "Qwen3-ForcedAligner-0.6B"))
).expanduser()
EXPLICIT_ALIGNER = os.getenv("ASR_ALIGNER", "").strip()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        return default


TIMESTAMPS_ENABLED = _env_bool("ASR_ENABLE_TIMESTAMPS", True)
SUBTITLE_MAX_CHARS = _env_int("ASR_SUBTITLE_MAX_CHARS", 42)
SUBTITLE_MAX_SECONDS = _env_float("ASR_SUBTITLE_MAX_SECONDS", 6.0, 0.1)
PSEUDO_STREAM_CHUNK_CHARS = _env_int("ASR_PSEUDO_STREAM_CHUNK_CHARS", 24)
PSEUDO_STREAM_CHUNK_DELAY_SECONDS = _env_float("ASR_PSEUDO_STREAM_CHUNK_DELAY_MS", 15.0, 0.0) / 1000.0


def _local_model_ready(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    return bool(weight_files)


def _resolve_checkpoint(
    explicit_path_or_id: str,
    local_dir: Path,
    remote_model_id: str,
    *,
    local_source: str,
) -> tuple[str, str]:
    if explicit_path_or_id:
        explicit_path = Path(explicit_path_or_id).expanduser()
        return explicit_path_or_id, "explicit-local" if explicit_path.exists() else "explicit-hub"
    if _local_model_ready(local_dir):
        return str(local_dir.resolve()), local_source
    return remote_model_id, "huggingface-fallback"


MODEL_LOAD_PATH, MODEL_SOURCE = _resolve_checkpoint(
    EXPLICIT_MODEL,
    LOCAL_MODEL_DIR,
    MODEL_NAME,
    local_source="modelscope-local",
)
ALIGNER_LOAD_PATH, ALIGNER_SOURCE = _resolve_checkpoint(
    EXPLICIT_ALIGNER,
    LOCAL_ALIGNER_DIR,
    ALIGNER_MODEL_NAME,
    local_source="modelscope-local",
)

DEVICE = os.getenv("ASR_DEVICE", "cuda:0")
DTYPE_NAME = os.getenv("ASR_DTYPE", "float16").lower()
MAX_BATCH = int(os.getenv("ASR_MAX_INFERENCE_BATCH_SIZE", "8"))
MAX_NEW_TOKENS = int(os.getenv("ASR_MAX_NEW_TOKENS", "256"))
ATTN_IMPLEMENTATION = os.getenv("ASR_ATTN_IMPLEMENTATION", "").strip()

# LID metadata is opt-in at the HTTP layer. Qwen itself auto-detects language
# whenever language=None. To preserve the old multilingual behavior, the default
# here is empty. If you want LID-off requests to force a language, set e.g.
# ASR_DEFAULT_LANGUAGE=Chinese.
DEFAULT_LANGUAGE = os.getenv("ASR_DEFAULT_LANGUAGE", "").strip()

_model: Qwen3ASRModel | None = None
_aligner: Qwen3ForcedAligner | None = None
_inference_lock = asyncio.Lock()

LANG_TO_ISO = {
    "Chinese": "zh",
    "English": "en",
    "Cantonese": "yue",
    "Arabic": "ar",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Portuguese": "pt",
    "Indonesian": "id",
    "Italian": "it",
    "Korean": "ko",
    "Russian": "ru",
    "Thai": "th",
    "Vietnamese": "vi",
    "Japanese": "ja",
    "Turkish": "tr",
    "Hindi": "hi",
    "Malay": "ms",
    "Dutch": "nl",
    "Swedish": "sv",
    "Danish": "da",
    "Finnish": "fi",
    "Polish": "pl",
    "Czech": "cs",
    "Filipino": "fil",
    "Persian": "fa",
    "Greek": "el",
    "Hungarian": "hu",
    "Macedonian": "mk",
    "Romanian": "ro",
}
ISO_TO_LANG = {value: key for key, value in LANG_TO_ISO.items()}

_ATTACH_TO_PREVIOUS = frozenset(",，、:：;；.。!！?？…\n")
_SENTENCE_ENDINGS = frozenset(".。!！?？…\n")


def _torch_dtype() -> torch.dtype:
    if DTYPE_NAME in ("float16", "fp16", "half"):
        return torch.float16
    if DTYPE_NAME in ("bfloat16", "bf16"):
        return torch.bfloat16
    if DTYPE_NAME in ("float32", "fp32"):
        return torch.float32
    raise RuntimeError(f"Unsupported ASR_DTYPE={DTYPE_NAME!r}")


def _load_model() -> Qwen3ASRModel:
    global _model
    if _model is not None:
        return _model

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This project is configured for NVIDIA GPU inference.")

    kwargs: dict[str, Any] = {
        "dtype": _torch_dtype(),
        "device_map": DEVICE,
        "max_inference_batch_size": MAX_BATCH,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    if ATTN_IMPLEMENTATION:
        kwargs["attn_implementation"] = ATTN_IMPLEMENTATION

    print(
        f"[startup] Loading {MODEL_LOAD_PATH} on {DEVICE}, dtype={DTYPE_NAME}, source={MODEL_SOURCE} ...",
        flush=True,
    )
    _model = Qwen3ASRModel.from_pretrained(MODEL_LOAD_PATH, **kwargs)
    print("[startup] ASR model loaded.", flush=True)
    return _model


def _load_forced_aligner() -> Qwen3ForcedAligner:
    """Load the timestamp model only when a request actually needs timestamps."""
    global _aligner
    if not TIMESTAMPS_ENABLED:
        raise RuntimeError("Timestamps are disabled. Set ASR_ENABLE_TIMESTAMPS=1 and restart the service.")
    if _aligner is not None:
        return _aligner

    kwargs: dict[str, Any] = {"dtype": _torch_dtype(), "device_map": DEVICE}
    if ATTN_IMPLEMENTATION:
        kwargs["attn_implementation"] = ATTN_IMPLEMENTATION

    print(
        f"[timestamps] Loading {ALIGNER_LOAD_PATH} on {DEVICE}, dtype={DTYPE_NAME}, source={ALIGNER_SOURCE} ...",
        flush=True,
    )
    _aligner = Qwen3ForcedAligner.from_pretrained(ALIGNER_LOAD_PATH, **kwargs)
    print("[timestamps] Forced aligner loaded.", flush=True)
    return _aligner


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="Qwen3-ASR-1.7B OpenAI-compatible service",
    version="1.3.0-timestamps-srt-pseudo-streaming",
    lifespan=lifespan,
)


def _parse_asr_options(raw: str | None) -> dict[str, Any]:
    if raw is None or raw.strip() == "":
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid asr_options JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="asr_options must be a JSON object")
    return value


def _bool_option(top_level: bool, options: dict[str, Any], key: str) -> bool:
    # Explicit top-level true always wins. Otherwise allow asr_options to set it.
    if top_level:
        return True
    value = options.get(key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_requested_language(language: str | None) -> str | None:
    if language is None or not language.strip():
        return None
    raw = language.strip()
    lower = raw.lower()
    if lower in ISO_TO_LANG:
        return ISO_TO_LANG[lower]
    # qwen-asr itself accepts/normalizes canonical language names.
    return raw


def _language_metadata(qwen_language: str) -> tuple[str, str]:
    """Return (ISO-ish code string, canonical Qwen language string)."""
    if not qwen_language:
        return "", ""
    names = [part.strip() for part in qwen_language.split(",") if part.strip()]
    codes = [LANG_TO_ISO.get(name, name.lower().replace(" ", "_")) for name in names]
    return ",".join(codes), ",".join(names)


def _coerce_option_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()]


def _normalize_timestamp_granularities(
    form_values: list[str] | None,
    bracket_values: list[str],
    options: dict[str, Any],
) -> set[str]:
    values = _coerce_option_values(form_values)
    values.extend(_coerce_option_values(bracket_values))
    values.extend(_coerce_option_values(options.get("timestamp_granularities")))
    granularities = {value.lower() for value in values if value}
    unsupported = granularities - {"word", "segment"}
    if unsupported:
        supported = "word, segment"
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported timestamp_granularities: {', '.join(sorted(unsupported))}. Supported values: {supported}",
        )
    return granularities


def _finite_nonnegative_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number >= 0 else 0.0


def _timestamp_rows(time_stamps: Any) -> list[dict[str, Any]]:
    """Normalize Qwen ForcedAlignResult items into JSON-safe timestamp rows."""
    if time_stamps is None:
        return []
    try:
        source_items = list(time_stamps)
    except TypeError:
        source_items = []

    rows: list[dict[str, Any]] = []
    for item in source_items:
        if isinstance(item, dict):
            token = item.get("text", "")
            start = item.get("start_time", item.get("start", 0))
            end = item.get("end_time", item.get("end", 0))
        else:
            token = getattr(item, "text", "")
            start = getattr(item, "start_time", getattr(item, "start", 0))
            end = getattr(item, "end_time", getattr(item, "end", 0))

        token_text = str(token or "")
        if not token_text:
            continue
        start_time = _finite_nonnegative_float(start)
        end_time = max(start_time, _finite_nonnegative_float(end))
        rows.append({"token": token_text, "start": start_time, "end": end_time})
    return rows


def _find_token(source: str, token: str, start_at: int) -> int:
    exact = source.find(token, start_at)
    if exact >= 0:
        return exact
    return source.lower().find(token.lower(), start_at)


def _build_timed_units(transcript_text: str, time_stamps: Any) -> list[dict[str, Any]]:
    """Attach punctuation from ASR text to the ForcedAligner token timings.

    The aligner intentionally strips most punctuation before alignment. This keeps
    SRT and verbose segment text readable while retaining the original token spans.
    """
    source = transcript_text or ""
    rows = _timestamp_rows(time_stamps)
    units: list[dict[str, Any]] = []
    cursor = 0

    for row in rows:
        token = row["token"]
        token_position = _find_token(source, token, cursor)
        display = token
        if token_position >= 0:
            prefix = source[cursor:token_position]
            token_end = token_position + len(token)
            source_token = source[token_position:token_end]
            if units and any(character in _ATTACH_TO_PREVIOUS for character in prefix):
                units[-1]["text"] += prefix
                display = source_token
            else:
                display = prefix + source_token
            cursor = token_end

        units.append(
            {
                "text": display,
                "start": row["start"],
                "end": row["end"],
            }
        )

    if units and cursor < len(source):
        units[-1]["text"] += source[cursor:]
    return units


def _ends_sentence(text: str) -> bool:
    return bool(text.rstrip()) and text.rstrip()[-1] in _SENTENCE_ENDINGS


def _build_timed_segments(transcript_text: str, time_stamps: Any) -> list[dict[str, Any]]:
    """Group aligned tokens into subtitle-friendly timed segments."""
    units = _build_timed_units(transcript_text, time_stamps)
    if not units:
        return []

    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for unit in units:
        if current is None:
            current = {"start": unit["start"], "end": unit["end"], "text": unit["text"]}
        else:
            current["end"] = max(current["end"], unit["end"])
            current["text"] += unit["text"]

        duration = current["end"] - current["start"]
        visible_length = len(current["text"].strip())
        close_on_sentence = _ends_sentence(current["text"]) and duration >= 0.75
        if close_on_sentence or visible_length >= SUBTITLE_MAX_CHARS or duration >= SUBTITLE_MAX_SECONDS:
            current["text"] = current["text"].strip()
            if current["text"]:
                segments.append(current)
            current = None

    if current is not None:
        current["text"] = current["text"].strip()
        if current["text"]:
            segments.append(current)

    return segments


def _apply_itn_to_segments(segments: list[dict[str, Any]], language_hint: str | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for segment in segments:
        updated = dict(segment)
        updated["text"] = apply_numeric_itn(updated["text"], language_hint=language_hint)
        normalized.append(updated)
    return normalized


def _verbose_segment_payload(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill the OpenAI verbose-JSON segment shape with unavailable metrics as 0."""
    payload: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        payload.append(
            {
                "id": index,
                "seek": int(round(segment["start"] * 100)),
                "start": round(segment["start"], 3),
                "end": round(max(segment["start"], segment["end"]), 3),
                "text": segment["text"],
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0,
            }
        )
    return payload


def _word_payload(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for unit in units:
        word = unit["text"].strip()
        if word:
            words.append(
                {
                    "word": word,
                    "start": round(unit["start"], 3),
                    "end": round(max(unit["start"], unit["end"]), 3),
                }
            )
    return words


def _timestamp_duration(units: list[dict[str, Any]]) -> float:
    return round(max((unit["end"] for unit in units), default=0.0), 3)


def _format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"


def _render_srt(segments: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = _finite_nonnegative_float(segment["start"])
        end = max(start + 0.001, _finite_nonnegative_float(segment["end"]))
        blocks.append(
            f"{index}\n{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}\n{segment['text']}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _sse_frame(event_name: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_name}\ndata: {data}\n\n"


async def _pseudo_stream_events(text: str) -> AsyncIterator[str]:
    """Emit OpenAI transcription SSE events after offline recognition is complete."""
    for start in range(0, len(text), PSEUDO_STREAM_CHUNK_CHARS):
        delta = text[start : start + PSEUDO_STREAM_CHUNK_CHARS]
        yield _sse_frame("transcript.text.delta", {"type": "transcript.text.delta", "delta": delta})
        if PSEUDO_STREAM_CHUNK_DELAY_SECONDS:
            await asyncio.sleep(PSEUDO_STREAM_CHUNK_DELAY_SECONDS)

    yield _sse_frame("transcript.text.done", {"type": "transcript.text.done", "text": text})
    yield "data: [DONE]\n\n"


def _transcribe_sync(
    path: str,
    prompt: str,
    forced_language: str | None,
    return_timestamps: bool,
):
    model = _load_model()
    if return_timestamps:
        # Qwen3ASRModel only checks this attribute at transcribe time, so adding
        # the lazy-loaded aligner is safe while requests are serialized below.
        model.forced_aligner = _load_forced_aligner()
    results = model.transcribe(
        audio=path,
        context=prompt or "",
        language=forced_language,
        return_time_stamps=return_timestamps,
    )
    if not results:
        raise RuntimeError("ASR model returned no result")
    return results[0]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_load_path": MODEL_LOAD_PATH,
        "model_source": MODEL_SOURCE,
        "local_model_dir": str(LOCAL_MODEL_DIR),
        "device": DEVICE,
        "dtype": DTYPE_NAME,
        "default_language": DEFAULT_LANGUAGE or None,
        "timestamps": {
            "enabled": TIMESTAMPS_ENABLED,
            "aligner": ALIGNER_MODEL_NAME,
            "aligner_load_path": ALIGNER_LOAD_PATH,
            "aligner_source": ALIGNER_SOURCE,
            "local_aligner_dir": str(LOCAL_ALIGNER_DIR),
            "aligner_loaded": _aligner is not None,
            "loading": "lazy-on-first-timestamp-request",
        },
        "features": {
            "enable_lid": "opt-in",
            "enable_itn": "opt-in; local rule-based ITN v4",
            "timestamps": "ForcedAligner-backed for verbose_json and srt",
            "srt": True,
            "stream": "OpenAI SSE pseudo-streaming; starts after offline recognition completes",
        },
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "local-qwen",
            }
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(MODEL_NAME),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    stream: bool = Form(False),
    timestamp_granularities: list[str] | None = Form(None),
    temperature: float = Form(0.0),  # accepted for OpenAI compatibility; Qwen uses deterministic decoding
    enable_lid: bool = Form(False),
    enable_itn: bool = Form(False),
    asr_options: str | None = Form(None),
):
    del model, temperature

    options = _parse_asr_options(asr_options)
    enable_lid = _bool_option(enable_lid, options, "enable_lid")
    enable_itn = _bool_option(enable_itn, options, "enable_itn")
    stream = _bool_option(stream, options, "stream")

    # The OpenAI Python SDK encodes arrays in multipart bodies as
    # timestamp_granularities[]. FastAPI handles repeated bare names itself.
    request_form = await request.form()
    bracket_granularities = [str(value) for value in request_form.getlist("timestamp_granularities[]")]
    granularities = _normalize_timestamp_granularities(timestamp_granularities, bracket_granularities, options)

    fmt = response_format.strip().lower()
    supported_formats = {"json", "verbose_json", "text", "srt"}
    if fmt not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail="Supported response_format values: json, verbose_json, text, srt",
        )
    if stream and fmt != "json":
        raise HTTPException(
            status_code=400,
            detail="stream=true only supports response_format=json",
        )
    if granularities and fmt != "verbose_json":
        raise HTTPException(
            status_code=400,
            detail="timestamp_granularities requires response_format=verbose_json",
        )

    return_timestamps = fmt in {"verbose_json", "srt"}
    if return_timestamps and not TIMESTAMPS_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Timestamp output is disabled. Set ASR_ENABLE_TIMESTAMPS=1 and restart the service.",
        )

    # Optional `language` can also be supplied inside asr_options, mirroring the
    # commercial Qwen API style.
    if (language is None or not language.strip()) and isinstance(options.get("language"), str):
        language = options["language"]

    requested_language = _normalize_requested_language(language)

    # Semantics:
    # - explicit language => force that language
    # - no language + enable_lid=true => pass None so Qwen performs LID
    # - no language + enable_lid=false => use ASR_DEFAULT_LANGUAGE if configured;
    #   otherwise preserve Qwen's normal language=None auto-detect behavior but
    #   do not expose LID metadata in the HTTP response.
    if requested_language is not None:
        forced_language = requested_language
    elif enable_lid:
        forced_language = None
    else:
        forced_language = _normalize_requested_language(DEFAULT_LANGUAGE) if DEFAULT_LANGUAGE else None

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    temp_path: str | None = None
    started = time.perf_counter()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        # Serialize requests on one GPU so concurrent HTTP calls do not fight for VRAM.
        async with _inference_lock:
            result = await asyncio.to_thread(
                _transcribe_sync,
                temp_path,
                prompt or "",
                forced_language,
                return_timestamps,
            )

        raw_text = result.text or ""
        language_hint = result.language or forced_language
        text = apply_numeric_itn(raw_text, language_hint=language_hint) if enable_itn else raw_text
        elapsed = time.perf_counter() - started
        lang_code, lang_name = _language_metadata(result.language or (forced_language or ""))

        timed_units: list[dict[str, Any]] = []
        timed_segments: list[dict[str, Any]] = []
        if return_timestamps:
            timed_units = _build_timed_units(raw_text, result.time_stamps)
            timed_segments = _build_timed_segments(raw_text, result.time_stamps)
            if enable_itn:
                # Alignment is calculated on raw ASR text. Applying ITN after
                # segment construction preserves each segment's time boundaries.
                timed_segments = _apply_itn_to_segments(timed_segments, language_hint)

        if stream:
            return StreamingResponse(
                _pseudo_stream_events(text),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-ASR-Stream-Mode": "pseudo",
                },
            )

        if fmt == "text":
            return PlainTextResponse(text)
        if fmt == "srt":
            return PlainTextResponse(_render_srt(timed_segments), media_type="application/x-subrip")

        payload: dict[str, Any] = {"text": text}
        if enable_lid:
            payload["language"] = lang_code
            payload["language_name"] = lang_name
        if fmt == "verbose_json":
            include_segments = not granularities or "segment" in granularities
            include_words = "word" in granularities
            duration = _timestamp_duration(timed_units)
            payload.update(
                {
                    # OpenAI's verbose response requires these fields even when
                    # callers did not request LID metadata explicitly.
                    "language": lang_code,
                    "duration": duration,
                    "segments": _verbose_segment_payload(timed_segments) if include_segments else None,
                    "words": _word_payload(timed_units) if include_words else None,
                    "usage": {"type": "duration", "seconds": duration},
                    "x_processing_seconds": round(elapsed, 4),
                    "x_enable_lid": enable_lid,
                    "x_enable_itn": enable_itn,
                    "x_timestamp_source": ALIGNER_MODEL_NAME,
                }
            )

        return JSONResponse(payload)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {type(exc).__name__}: {exc}") from exc
    finally:
        try:
            await file.close()
        except Exception:
            pass
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
