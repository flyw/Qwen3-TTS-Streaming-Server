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

Thanks to the internal optimization of the generation loop (intercepting tokens via forward hooks), the server achieves industry-leading response times.

### Test Environment: NVIDIA RTX 4070
**Input**: ~50 characters/words.

```text
Sending Payload: {"text":"[INPUT_TEXT]","language":"Chinese","stream":true}
[T+245ms] First chunk received (2560 bytes)
[T+311ms] Second chunk received
[T+371ms] Third chunk received
...
[T+1405ms] Stream completed
```

**Key Metrics**:
- **Time to First Chunk (TTFC)**: **~245ms**
- **Chunk Interval**: ~60ms
- **Stability**: Consistent throughput even with complex linguistic structures.

---

## Features

* **Sub-300ms Latency**: First audio packet is delivered while the model is still generating the rest of the sentence.
* **High-Fidelity Voice Cloning**: Supports rapid 3-second voice cloning with the 12Hz-1.7B-Base model.
* **Stream & Non-Stream Support**: Flexible API for both real-time dialogue and batch generation.
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
python server.py \
  --model-path ./Qwen3-TTS-12Hz-1.7B-Base \
  --ref-audio-file reference.wav \
  --ref-text-file reference.txt \
  --port 9000
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
  "temperature": 0.5
}
```

**Response**: A `text/event-stream` returning JSON chunks:
- `{"type": "start"}`: Signals the start of generation.
- `{"type": "audio", "data": "BASE64_WAV_CHUNK", "index": 1}`: Audio data chunk (WAV format).
- `{"type": "done"}`: Signals completion.

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
