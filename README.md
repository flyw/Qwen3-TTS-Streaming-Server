# Qwen3-TTS Streaming Server

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/qwen3_tts_logo.png" width="400"/>
</p>

<p align="center">
&nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen3-tts">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/collections/Qwen/Qwen3-TTS">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://qwen.ai/blog?id=qwen3tts-0115">Blog</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2601.15621">Paper</a>&nbsp&nbsp
</p>

## Overview

**Qwen3-TTS Streaming Server** is a high-performance, production-ready FastAPI wrapper for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). This project forks the original repository to provide a standardized, easy-to-deploy streaming service with enhanced features for real-time applications.

### What's New in This Fork?
- **High-Performance FastAPI Server**: A dedicated server implementation (`server.py`) optimized for low-latency concurrency.
- **Precision SSE Streaming**: Implements **Server-Sent Events (SSE)** for real-time audio token streaming.
- **Sliding Window Audio Reconstruction**: Uses an advanced sliding window algorithm to ensure smooth, high-quality audio output during streaming.
- **Production-Ready Configuration**: Full support for Environment Variables and CLI arguments for flexible deployment (Model path, Host, Port, Reference Audio/Text).
- **Comprehensive Client Support**: Includes detailed integration guides and examples for Python and JavaScript (Web Audio API).
- **Cleaned & Globalized**: Fully localized comments and logs in English, ready for the global open-source community.

---

## Features

* **Extreme Low-Latency**: Immediate first-packet delivery (~700ms-1300ms depending on text length).
* **High-Fidelity Voice Cloning**: Supports 3-second rapid voice cloning with superior similarity.
* **Stream & Non-Stream Support**: Standardized API for various integration scenarios.
* **Automatic Language Detection**: Supports 10+ languages (CN, EN, JP, KR, DE, FR, RU, PT, ES, IT).

---

## Quickstart

### 1. Environment Setup

We recommend using Python 3.12 with a clean environment:

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts uvicorn fastapi soundfile
# Optional: FlashAttention 2 for performance
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

Run the server with your desired configuration:

```bash
python server.py \
  --model-path ./Qwen3-TTS-12Hz-1.7B-Base \
  --ref-audio-file your_voice.wav \
  --ref-text-file your_voice_transcript.txt \
  --port 9000
```

---

## API Reference

### 1. Streaming TTS (SSE)
**Endpoint**: `POST /tts/stream`

**Request Body**:
```json
{
  "text": "Hello, this is a real-time streaming test.",
  "language": "English",
  "temperature": 0.5,
  "max_new_tokens": 2048
}
```

**Response**: A `text/event-stream` returning JSON chunks:
- `{"type": "start"}`: Signals the start of generation.
- `{"type": "audio", "data": "BASE64_WAV_CHUNK", "index": 1}`: Audio data chunk (WAV format).
- `{"type": "done"}`: Signals completion.

### 2. Health Check
**Endpoint**: `GET /health`

---

## Configuration

| CLI Argument | Environment Variable | Default | Description |
|--------------|----------------------|---------|-------------|
| `--model-path` | `MODEL_PATH` | `./Qwen3-TTS-12Hz-1.7B-Base` | Path to model weights |
| `--ref-audio-file` | `REF_AUDIO_PATH` | `None` | Default reference audio for cloning |
| `--ref-text-file` | `REF_TEXT` | `None` | Transcript of the reference audio |
| `--host` | `HOST` | `0.0.0.0` | Server host |
| `--port` | `PORT` | `9000` | Server port |

---

## Client Integration Guide

### JavaScript / Frontend (Web Audio API)

Receiving audio chunks via SSE requires careful handling of the WAV header and sample rate.

```javascript
// Correct way to play streaming audio chunks in the browser
async function playWavChunk(base64Data) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    const binary = atob(base64Data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    
    // Decode complete WAV data (including the 44-byte header provided by the server)
    const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.start(0);
}
```

**Common Pitfalls**:
- **Noise only**: Usually caused by skipping the 44-byte WAV header or using the wrong sample rate (Fixed: Use `24000Hz`).
- **Popping sounds**: Ensure you are using the `decodeAudioData` method which handles the underlying PCM format correctly.

---

## Credits

Special thanks to the **Qwen Team** for developing and open-sourcing the revolutionary Qwen3-TTS models. This project is a community-driven extension to improve accessibility and deployment efficiency.

## License

This project follows the original [License](LICENSE). Please adhere to the terms provided by the Qwen team.
