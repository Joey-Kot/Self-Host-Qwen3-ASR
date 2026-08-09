# Self-Host Qwen3-ASR

**English** | [简体中文](README_ZH.md)

A self-hosted Qwen3-ASR service for Windows, built with `Qwen/Qwen3-ASR-1.7B`, FastAPI, and CUDA FP16. It provides an OpenAI-compatible speech transcription API with timestamps, SRT subtitles, SSE pseudo-streaming, and optional LID and ITN.

Default service URL: `http://127.0.0.1:8000`

## 1. Supported endpoints, response formats, and delivery modes

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/audio/transcriptions` | Upload and transcribe an audio file through an OpenAI Audio Transcriptions-compatible API |
| `GET` | `/v1/models` | Return the currently available model list |
| `GET` | `/health` | Return service, model, device, timestamp model, and feature status |

The transcription endpoint accepts `multipart/form-data`. Common fields are listed below:

| Field | Default | Description |
| --- | --- | --- |
| `file` | Required | Audio file to transcribe |
| `model` | `Qwen/Qwen3-ASR-1.7B` | Model field retained for OpenAI API compatibility |
| `language` | Empty | Force a language such as `zh`, `en`, `Chinese`, or `English` |
| `prompt` | Empty | Context prompt passed to the model |
| `response_format` | `json` | One of `json`, `verbose_json`, `text`, or `srt` |
| `stream` | `false` | Return SSE pseudo-streaming events when `true`; only supported with `response_format=json` |
| `timestamp_granularities[]` | Empty | Timestamp granularity for `verbose_json`; `segment` and `word` may be supplied repeatedly |
| `enable_lid` | `false` | Include language identification metadata in the response |
| `enable_itn` | `false` | Enable local rule-based inverse text normalization |
| `asr_options` | Empty | JSON object supporting `language`, `enable_lid`, `enable_itn`, `stream`, and `timestamp_granularities` |
| `temperature` | `0` | Accepted for OpenAI API compatibility; Qwen3-ASR uses deterministic decoding |

### Response formats

| `response_format` | Content-Type | Output |
| --- | --- | --- |
| `json` | `application/json` | Default format. Returns `text` and, when LID is enabled, `language` and `language_name` |
| `verbose_json` | `application/json` | Returns language, duration, segment/word timestamps, processing time, and other details |
| `text` | `text/plain` | Returns only the transcription text |
| `srt` | `application/x-subrip` | Returns SRT subtitles with timestamps |

`verbose_json` and `srt` use `Qwen/Qwen3-ForcedAligner-0.6B` to generate aligned timestamps. The ForcedAligner is loaded lazily on the first request that needs timestamps and consumes additional VRAM.

When `timestamp_granularities[]` is omitted, `verbose_json` includes `segments` by default. To request both segment-level and word-level timestamps, supply:

```text
timestamp_granularities[]=segment
timestamp_granularities[]=word
```

Standard JSON response:

```json
{
  "text": "Hello, and welcome to Qwen3-ASR."
}
```

With LID enabled:

```json
{
  "text": "Hello, and welcome to Qwen3-ASR.",
  "language": "en",
  "language_name": "English"
}
```

### Delivery modes

The service supports two delivery modes:

1. **Standard response**: the default. The service transcribes the complete audio file and returns JSON, plain text, or SRT in one response.
2. **SSE pseudo-streaming**: set `stream=true`. The service still completes offline transcription first, then splits the final text into multiple SSE events.

Pseudo-streaming is compatible with the OpenAI transcription event format, but it is not real-time streaming and does not reduce time to first text:

```text
event: transcript.text.delta
data: {"type":"transcript.text.delta","delta":"..."}

event: transcript.text.done
data: {"type":"transcript.text.done","text":"..."}

data: [DONE]
```

### Request examples

Standard transcription:

```bat
curl.exe http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "model=Qwen/Qwen3-ASR-1.7B" ^
  -F "file=@test.wav"
```

Return word-level and segment-level timestamps:

```bat
curl.exe http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "model=Qwen/Qwen3-ASR-1.7B" ^
  -F "response_format=verbose_json" ^
  -F "timestamp_granularities[]=segment" ^
  -F "timestamp_granularities[]=word" ^
  -F "file=@test.wav"
```

Generate SRT subtitles:

```bat
curl.exe http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "model=Qwen/Qwen3-ASR-1.7B" ^
  -F "response_format=srt" ^
  -F "file=@test.wav" ^
  -o subtitles.srt
```

Use SSE pseudo-streaming:

```bat
curl.exe -N http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "model=Qwen/Qwen3-ASR-1.7B" ^
  -F "stream=true" ^
  -F "file=@test.wav"
```

The OpenAI Python SDK is also supported. See [`client_example.py`](client_example.py).

## 2. Deployment

### Requirements

- Windows 10/11
- Python 3.12 with a working `py -3.12` launcher
- An NVIDIA GPU with CUDA support and a recent graphics driver
- Sufficient disk space and VRAM; timestamp output requires the ForcedAligner to be loaded alongside the ASR model

The current setup script installs the CUDA 12.8 build of PyTorch. Environment setup downloads several large packages and usually takes longer than a typical Python dependency installation.

### Step 1: Install the environment and dependencies

Double-click or run:

```bat
setup.bat
```

The script:

- Creates a project-local `.venv` virtual environment.
- Installs CUDA-enabled PyTorch, Qwen ASR, FastAPI, ModelScope, and the other dependencies.
- Verifies the runtime environment.
- Runs the ITN regression tests.

The current `setup.bat` also attempts to download the models after installing the dependencies. You can still run the next step regardless of whether that attempt succeeds: complete models are skipped, while missing or incomplete models are downloaded again.

### Step 2: Download both models

Run:

```bat
download_model.bat
```

By default, the script downloads and verifies both models through ModelScope:

| Model | Purpose | Local directory |
| --- | --- | --- |
| `Qwen/Qwen3-ASR-1.7B` | Speech recognition | `models\Qwen3-ASR-1.7B` |
| `Qwen/Qwen3-ForcedAligner-0.6B` | Word/segment timestamps and SRT | `models\Qwen3-ForcedAligner-0.6B` |

ModelScope is generally fast on networks in Mainland China. Disabling proxies while downloading the models is recommended. The script is safe to run repeatedly and skips models that are already complete.

To use Hugging Face instead, run:

```bat
download_model.bat HuggingFace
```

### Step 3: Start the service in the background

Double-click:

```text
start_silent.vbs
```

This launches `start.bat` in a hidden background window. The first startup may take some time while the ASR model is loaded. By default, the service listens at:

```text
http://127.0.0.1:8000
```

Use the health endpoint to verify that the service is ready:

```bat
curl.exe http://127.0.0.1:8000/health
```

When the local model is loaded normally, `model_source` in the health response should be `modelscope-local`.

### Changing the port and startup parameters

Edit `start.bat` to change the port, listening address, model paths, device, or other startup settings.

The default launch command is at the end of the file:

```bat
"%CD%\.venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
```

For example, replace `--port 8000` with `--port 9000` to change the port. Common environment variables include:

| Variable | Default | Description |
| --- | --- | --- |
| `ASR_DEVICE` | `cuda:0` | Inference device |
| `ASR_DTYPE` | `float16` | Inference precision: `float16`, `bfloat16`, or `float32` |
| `ASR_MAX_INFERENCE_BATCH_SIZE` | `8` | Maximum model inference batch size |
| `ASR_MAX_NEW_TOKENS` | `256` | Maximum number of generated tokens |
| `ASR_ENABLE_TIMESTAMPS` | `1` | Enable `verbose_json` and SRT timestamp output; related requests return HTTP 503 when disabled |
| `ASR_DEFAULT_LANGUAGE` | Empty | Forced default language when no language is supplied and LID metadata is not requested |
| `ASR_LOCAL_MODEL_DIR` | `models\Qwen3-ASR-1.7B` | Local ASR model directory |
| `ASR_LOCAL_ALIGNER_DIR` | `models\Qwen3-ForcedAligner-0.6B` | Local ForcedAligner model directory |
| `ASR_SUBTITLE_MAX_CHARS` | `42` | Maximum number of characters in one subtitle segment |
| `ASR_SUBTITLE_MAX_SECONDS` | `6.0` | Maximum duration of one subtitle segment |

The project uses one worker by default and serializes inference requests on a single GPU to reduce VRAM contention.

## 3. LID and ITN

### LID: language identification metadata

LID identifies the language spoken in the audio. The `enable_lid` request parameter is disabled by default.

Here, “disabled” primarily means that language metadata is omitted from standard JSON responses. When both `language` and `ASR_DEFAULT_LANGUAGE` are empty, Qwen3-ASR still detects the language internally.

| Request | Model behavior | Standard JSON response |
| --- | --- | --- |
| No `language`, `enable_lid=false` | Use `ASR_DEFAULT_LANGUAGE`; if empty, let Qwen detect the language | Return only `text` |
| No `language`, `enable_lid=true` | Let Qwen detect the language | Return `text`, `language`, and `language_name` |
| Supply `language=zh/en/...` | Force the requested language | Add language metadata only when `enable_lid=true` |

For OpenAI detailed-response compatibility, `verbose_json` always includes the `language` field, regardless of `enable_lid`.

Enable LID:

```bat
curl.exe http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "enable_lid=true" ^
  -F "file=@test.wav"
```

### ITN: inverse text normalization

ITN converts spoken-form numbers and measurements in the transcript into written forms that are easier to read, store, and process. The `enable_itn` request parameter is disabled by default.

This project uses a local, deterministic, rule-based ITN v4 implementation. It runs after recognition and does not change ASR model inference.

| Type | Input example | Output example |
| --- | --- | --- |
| Numbers and decimals | `三十八`, `一点七` | `38`, `1.7` |
| Ratios | `百分之三十八`, `千分之一` | `38%`, `1‰` |
| Currency | `三十八美元` | `USD 38` |
| Units | `三十摄氏度`, `三公里` | `30°C`, `3 km` |
| Dates | `二零二六年八月八日` | `2026-08-08` |
| Clock times | `下午三点半` | `15:30` |
| Durations | `三小时二十分钟五秒` | `3 h 20 min 5 s` |

ITN supports common Chinese and English expressions, including some mixed Chinese-English text. It does not process or infer:

- Phone numbers or general-purpose ordinals.
- Conversion of relative dates into absolute dates.
- Time-zone conversion or calendar arithmetic.
- Currency subunits, exchange rates, arithmetic, or physical unit conversion.
- Punctuation insertion or semantic rewriting.

Enable ITN:

```bat
curl.exe http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "enable_itn=true" ^
  -F "file=@test.wav"
```

Enable both LID and ITN:

```bat
curl.exe http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "enable_lid=true" ^
  -F "enable_itn=true" ^
  -F "file=@test.wav"
```

Alternatively, pass the options through `asr_options`:

```bat
curl.exe http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "asr_options={\"enable_lid\":true,\"enable_itn\":true}" ^
  -F "file=@test.wav"
```

Timestamps are always generated from the raw ASR text. When ITN is enabled, the service determines time boundaries first and then normalizes each subtitle segment, so the timeline is not recalculated or shifted.

## 4. License

This project is licensed under the [MIT License](LICENSE). See [`LICENSE`](LICENSE) for the full terms and [`NOTICE`](NOTICE) for the project notice.

Third-party components retain their original licenses and notices.
