# Self-Host Qwen3-ASR

[English](README.md) | **简体中文**

面向 Windows 的本地 Qwen3-ASR 服务，使用 `Qwen/Qwen3-ASR-1.7B`、FastAPI 和 CUDA FP16，提供 OpenAI 兼容的语音转写接口，并支持时间戳、SRT 字幕、SSE 伪流式输出，以及可选的 LID 和 ITN。

默认服务地址：`http://127.0.0.1:8000`

## 1. 支持的端点、输出格式与输出方式

### 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/v1/audio/transcriptions` | 上传音频并转写，兼容 OpenAI Audio Transcriptions API |
| `GET` | `/v1/models` | 返回当前可用模型列表 |
| `GET` | `/health` | 返回服务、模型、设备、时间戳模型和功能状态 |

转写接口使用 `multipart/form-data`。常用字段如下：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `file` | 必填 | 待转写的音频文件 |
| `model` | `Qwen/Qwen3-ASR-1.7B` | 为兼容 OpenAI API 保留的模型字段 |
| `language` | 空 | 指定语言，例如 `zh`、`en`、`Chinese`、`English`；填写后会强制使用该语言 |
| `prompt` | 空 | 提供给模型的上下文提示 |
| `response_format` | `json` | 可选 `json`、`verbose_json`、`text`、`srt` |
| `stream` | `false` | 设为 `true` 时返回 SSE 伪流式事件，仅支持 `response_format=json` |
| `timestamp_granularities[]` | 空 | `verbose_json` 的时间戳粒度，可重复传入 `segment` 和 `word` |
| `enable_lid` | `false` | 是否在响应中返回语言识别信息 |
| `enable_itn` | `false` | 是否启用本地规则式逆文本标准化 |
| `asr_options` | 空 | JSON 对象，可传入 `language`、`enable_lid`、`enable_itn`、`stream` 和 `timestamp_granularities` |
| `temperature` | `0` | 为兼容 OpenAI API 接收；Qwen3-ASR 使用确定性解码 |

### 输出格式

| `response_format` | Content-Type | 输出内容 |
| --- | --- | --- |
| `json` | `application/json` | 默认格式，返回 `text`；启用 LID 后增加 `language` 和 `language_name` |
| `verbose_json` | `application/json` | 返回语言、时长、分段/词级时间戳、处理耗时等详细信息 |
| `text` | `text/plain` | 仅返回转写文本 |
| `srt` | `application/x-subrip` | 返回带时间轴的 SRT 字幕 |

`verbose_json` 和 `srt` 会使用 `Qwen/Qwen3-ForcedAligner-0.6B` 生成真实对齐时间戳。ForcedAligner 在第一次需要时间戳的请求中延迟加载，并会占用额外显存。

未传入 `timestamp_granularities[]` 时，`verbose_json` 默认返回 `segments`。如需词级时间戳，可同时传入：

```text
timestamp_granularities[]=segment
timestamp_granularities[]=word
```

普通 JSON 响应示例：

```json
{
  "text": "你好，欢迎使用 Qwen3-ASR。"
}
```

启用 LID 后：

```json
{
  "text": "你好，欢迎使用 Qwen3-ASR。",
  "language": "zh",
  "language_name": "Chinese"
}
```

### 输出方式

服务支持两种输出方式：

1. **普通响应**：默认方式。服务完成整段音频识别后，一次性返回 JSON、纯文本或 SRT。
2. **SSE 伪流式响应**：设置 `stream=true`。服务仍会先完成整段音频的离线识别，再把最终文本拆成多个 SSE 事件发送。

伪流式输出兼容 OpenAI 的转写流事件格式，但不是真正的实时流式识别，也不会降低首字等待时间：

```text
event: transcript.text.delta
data: {"type":"transcript.text.delta","delta":"..."}

event: transcript.text.done
data: {"type":"transcript.text.done","text":"..."}

data: [DONE]
```

### 调用示例

普通转写：

```bat
curl.exe http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "model=Qwen/Qwen3-ASR-1.7B" ^
  -F "file=@test.wav"
```

返回词级和分段时间戳：

```bat
curl.exe http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "model=Qwen/Qwen3-ASR-1.7B" ^
  -F "response_format=verbose_json" ^
  -F "timestamp_granularities[]=segment" ^
  -F "timestamp_granularities[]=word" ^
  -F "file=@test.wav"
```

生成 SRT 字幕：

```bat
curl.exe http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "model=Qwen/Qwen3-ASR-1.7B" ^
  -F "response_format=srt" ^
  -F "file=@test.wav" ^
  -o subtitles.srt
```

SSE 伪流式输出：

```bat
curl.exe -N http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "model=Qwen/Qwen3-ASR-1.7B" ^
  -F "stream=true" ^
  -F "file=@test.wav"
```

也可以直接使用 OpenAI Python SDK，示例见 [`client_example.py`](client_example.py)。

## 2. 部署方式

### 环境要求

- Windows 10/11
- Python 3.12，并确保 `py -3.12` 可以正常运行
- 支持 CUDA 的 NVIDIA 显卡及较新的显卡驱动
- 足够的磁盘空间和显存；时间戳输出还需要同时加载 ForcedAligner

当前安装脚本会安装 CUDA 12.8 版本的 PyTorch。环境和依赖安装包含较大的软件包，耗时通常比普通 Python 项目更长，请耐心等待。

### 第一步：安装环境和依赖

双击或在命令提示符中运行：

```bat
setup.bat
```

脚本会：

- 创建项目内的 `.venv` 虚拟环境；
- 安装 CUDA 版 PyTorch、Qwen ASR、FastAPI、ModelScope 等依赖；
- 检查运行环境；
- 运行 ITN 回归测试。

当前 `setup.bat` 在依赖安装完成后也会尝试下载模型。无论这一步是否成功，都可以继续运行下一步；已完整下载的模型会被自动跳过，缺失或不完整的模型会重新下载。

### 第二步：下载两个模型

运行：

```bat
download_model.bat
```

脚本默认通过 ModelScope 下载并检查以下两个模型：

| 模型 | 用途 | 本地目录 |
| --- | --- | --- |
| `Qwen/Qwen3-ASR-1.7B` | 语音识别 | `models\Qwen3-ASR-1.7B` |
| `Qwen/Qwen3-ForcedAligner-0.6B` | 词级/分段时间戳和 SRT | `models\Qwen3-ForcedAligner-0.6B` |

ModelScope 在国内网络中通常下载很快，建议下载模型时不要开启代理。脚本可以重复运行，已完成的模型会直接跳过。

如确实需要改用 Hugging Face，可运行：

```bat
download_model.bat HuggingFace
```

### 第三步：后台启动服务

双击：

```text
start_silent.vbs
```

它会隐藏窗口并在后台调用 `start.bat`。首次启动需要加载 ASR 模型，可能需要等待一段时间。服务默认监听：

```text
http://127.0.0.1:8000
```

可通过健康检查确认服务是否已启动：

```bat
curl.exe http://127.0.0.1:8000/health
```

正常使用本地模型时，健康检查中的 `model_source` 应为 `modelscope-local`。

### 修改端口和启动参数

如需修改端口、监听地址、模型路径、设备或其他启动参数，请编辑 `start.bat`。

默认启动命令位于文件末尾：

```bat
"%CD%\.venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
```

例如，将 `--port 8000` 改为 `--port 9000` 即可修改端口。常用环境变量如下：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `ASR_DEVICE` | `cuda:0` | 推理设备 |
| `ASR_DTYPE` | `float16` | 推理精度，可选 `float16`、`bfloat16`、`float32` |
| `ASR_MAX_INFERENCE_BATCH_SIZE` | `8` | 模型推理批大小上限 |
| `ASR_MAX_NEW_TOKENS` | `256` | 最大生成 token 数 |
| `ASR_ENABLE_TIMESTAMPS` | `1` | 是否允许 `verbose_json` 和 SRT 时间戳输出；关闭后相关请求返回 HTTP 503 |
| `ASR_DEFAULT_LANGUAGE` | 空 | 未显式指定语言且未请求 LID 元数据时使用的强制默认语言 |
| `ASR_LOCAL_MODEL_DIR` | `models\Qwen3-ASR-1.7B` | ASR 模型目录 |
| `ASR_LOCAL_ALIGNER_DIR` | `models\Qwen3-ForcedAligner-0.6B` | ForcedAligner 模型目录 |
| `ASR_SUBTITLE_MAX_CHARS` | `42` | 单个字幕分段的最大字符数 |
| `ASR_SUBTITLE_MAX_SECONDS` | `6.0` | 单个字幕分段的最长持续时间 |

项目默认使用单个 worker，并串行执行同一张 GPU 上的推理请求，以减少并发抢占显存的问题。

## 3. LID 和 ITN

### LID：语言识别信息

LID（Language Identification）用于识别音频语言。请求参数 `enable_lid` 默认关闭。

这里的“关闭”主要表示不在普通 JSON 响应中返回语言元数据：当 `language` 和 `ASR_DEFAULT_LANGUAGE` 都为空时，Qwen3-ASR 仍会在模型内部自动判断语言。

| 请求方式 | 模型行为 | 普通 JSON 响应 |
| --- | --- | --- |
| 不传 `language`，`enable_lid=false` | 使用 `ASR_DEFAULT_LANGUAGE`；若为空则由 Qwen 自动判断 | 只返回 `text` |
| 不传 `language`，`enable_lid=true` | 由 Qwen 自动判断语言 | 返回 `text`、`language`、`language_name` |
| 传入 `language=zh/en/...` | 强制使用指定语言 | 仅在 `enable_lid=true` 时附加语言元数据 |

`verbose_json` 为兼容 OpenAI 的详细响应结构，无论 `enable_lid` 是否开启，都会包含 `language` 字段。

启用 LID：

```bat
curl.exe http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "enable_lid=true" ^
  -F "file=@test.wav"
```

### ITN：逆文本标准化

ITN（Inverse Text Normalization）把语音识别结果中的口语化数字和量词转换为更适合阅读、存储和后续处理的书面格式。请求参数 `enable_itn` 默认关闭。

本项目使用本地、确定性、规则式 ITN。只处理识别完成后的文本，不会改变 ASR 模型推理。

| 类型 | 输入示例 | 输出示例 |
| --- | --- | --- |
| 数字和小数 | `三十八`、`一点七` | `38`、`1.7` |
| 比例 | `百分之三十八`、`千分之一` | `38%`、`1‰` |
| 货币 | `三十八美元` | `USD 38` |
| 单位 | `三十摄氏度`、`三公里` | `30°C`、`3 km` |
| 日期 | `二零二六年八月八日` | `2026-08-08` |
| 时间 | `下午三点半` | `15:30` |
| 时长 | `三小时二十分钟五秒` | `3 h 20 min 5 s` |

ITN 同时支持常见中文和英文表达，以及部分中英文混合文本。它不会处理或推断以下内容：

- 电话号码和一般序数；
- 相对日期到绝对日期的换算；
- 时区转换和日历运算；
- 货币子单位、汇率、算术或物理单位换算；
- 标点补全或语义改写。

启用 ITN：

```bat
curl.exe http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "enable_itn=true" ^
  -F "file=@test.wav"
```

也可以同时启用 LID 和 ITN：

```bat
curl.exe http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "enable_lid=true" ^
  -F "enable_itn=true" ^
  -F "file=@test.wav"
```

或者通过 `asr_options` 传入：

```bat
curl.exe http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "asr_options={\"enable_lid\":true,\"enable_itn\":true}" ^
  -F "file=@test.wav"
```

时间戳始终基于原始 ASR 文本生成。启用 ITN 时，服务会先确定时间边界，再标准化每个字幕分段的文本，因此不会重新计算或移动时间轴。

## 4. 许可证

本项目采用 [MIT License](LICENSE)。完整条款见 [`LICENSE`](LICENSE)，项目声明见 [`NOTICE`](NOTICE)。

第三方组件保留各自原有的许可证与声明。
