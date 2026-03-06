# Qwen3-TTS Streaming Server

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/qwen3_tts_logo.png" width="400"/>
</p>

<p align="center">
&nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen3-tts">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/collections/Qwen/Qwen3-TTS">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://qwen.ai/blog?id=qwen3tts-0115">Blog</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2601.15621">Paper</a>&nbsp&nbsp
</p>

## Overview

**Qwen3-TTS Streaming Server** is a high-performance, production-ready FastAPI wrapper for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). This project focuses on delivering ultra-low latency audio streaming for real-time human-computer interaction.

### What's New in This Fork?
- **Extreme Performance Optimization**: Utilizes a specialized monkey-patching technique to intercept model forward passes, enabling the delivery of the first audio chunk almost instantly.
- **High-Performance FastAPI Server**: Optimized for concurrent requests and low-latency throughput.
- **Precision SSE Streaming**: Implements Server-Sent Events (SSE) for reliable real-time audio token delivery.
- **Sliding Window Audio Reconstruction**: An advanced algorithm ensures seamless audio stitching and high-quality output during streaming.
- **Production Configuration**: Full support for CLI arguments and Environment Variables (Model path, Host, Port, Reference Audio).

---

## Performance Benchmark

The following metrics were captured using the included `webui.html` on a system equipped with an **NVIDIA RTX 4070 GPU** with **chunk_size: 1** (lowest latency mode) enabled.

### Real-World Metrics:
- **Time to First Token (TTFT)**: **~426ms** (From request sent to audio starting)
- **Average Chunk Interval**: **~308ms** (Time between receiving streaming packets)
- **Throughput**: Fast delivery suitable for real-time interaction.

---

## Features

* **Sub-500ms Latency**: Industry-leading response time for real-time voice interaction.
* **X-Vector Only Mode**: Support for pure speaker embedding cloning, which **100% eliminates prompt leakage** (no more hallucinating or repeating reference text).
* **Configurable Streaming Buffer**: Prevent audio stuttering during network fluctuations by buffering tokens before sending (`chunk_size`).
* **High-Fidelity Voice Cloning**: Supports rapid 3-second voice cloning with the 12Hz-1.7B-Base model.
* **Multi-Language Support**: Native support for CN, EN, JP, KR, DE, FR, RU, PT, ES, IT.

---

## Quickstart

### 1. Environment Setup

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts uvicorn fastapi soundfile
# Highly recommended: FlashAttention 2
pip install -U flash-attn --no-build-isolation
```

### 2. Model Preparation

Download the `Qwen3-TTS-12Hz-1.7B-Base` model (or other variants) to your local directory:

```bash
# Example using ModelScope
pip install -U modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir ./Qwen3-TTS-12Hz-1.7B-Base
```

### 3. Launch the Server

```bash
# Recommended: X-Vector mode is enabled by default in server.py to prevent prompt leakage
# You only need the reference audio file. No transcript required.
python server.py \
  --model-path ./Qwen3-TTS-12Hz-1.7B-Base \
  --ref-audio-file reference.wav \
  --port 9000 \
  --chunk-size 1
```

---

## API Reference

### Streaming TTS (SSE)
**Endpoint**: `POST /tts/stream`

**Request Body**:
```json
{
  "text": "Hello world.",
  "language": "English",
  "temperature": 0.5,
  "chunk_size": 1
}
```
*Note: We have optimized the server to use **X-Vector mode** by default. This ensures that the model only clones the speaker's voice identity without being distracted by the content of the reference audio, effectively eliminating "hallucinations" (repeating words from the reference clip). This mode does not require a reference transcript.*

**Response**: A `text/event-stream` returning JSON chunks:
- `{"type": "start"}`: Signals the start of generation.
- `{"type": "audio", "data": "BASE64_WAV_CHUNK", "index": 1, "chunk_len": 1}`: Audio data chunk (WAV format).
- `{"type": "done"}`: Signals completion.

---

## Configuration

| CLI Argument | Environment Variable | Default | Description |
|--------------|----------------------|---------|-------------|
| `--model-path` | `MODEL_PATH` | `./Qwen3-TTS-12Hz-1.7B-Base` | Path to model weights |
| `--ref-audio-file` | `REF_AUDIO_PATH` | `None` | Path to reference audio for cloning (Required) |
| `--host` | `HOST` | `0.0.0.0` | Server host |
| `--port` | `PORT` | `9000` | Server port |
| `--chunk-size` | *None* | `1` | Global default for tokens to buffer before sending |

---

## WebUI Testing

A ready-to-use HTML client (`webui.html`) is included to test the streaming API. It utilizes the Web Audio API to seamlessly stitch and play incoming chunks.

1. Start the server (e.g., on port 9000).
2. Open `webui.html` in any modern web browser.
3. Observe the **Time to First Token (TTFT)** and playback smoothness. Adjust the `chunk_size` in the server startup or HTML payload if you experience audio stuttering.

---

## Client Integration Guide

### JavaScript / Frontend (Web Audio API)

```javascript
async function playWavChunk(base64Data) {
    const audioContext = new AudioContext({ sampleRate: 24000 });
    const binary = atob(base64Data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    
    const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.start(0);
}
```

---

## Credits

Developed based on the excellent work of the **Qwen Team**. This extension aims to provide the community with a high-speed serving alternative for real-time AI applications.

## License

Adheres to the original [License](LICENSE) provided by the Qwen team.
