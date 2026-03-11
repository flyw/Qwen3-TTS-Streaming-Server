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
- **Raw Binary Streaming**: Replaced SSE/Base64 with raw **PCM 16-bit** binary streaming, reducing bandwidth overhead by **~33%** and lowering client-side CPU usage.
- **Sliding Window Audio Reconstruction**: An advanced algorithm ensures seamless audio stitching and high-quality output during streaming.
- **Production Configuration**: Full support for CLI arguments and Environment Variables (Model path, Host, Port, Reference Audio).
- **Batch Decoding**: Configurable `chunk-size` on the server to balance RTF performance and real-time feel.

---

## Performance Benchmark

The following metrics were captured using the included `webui.html` on a system equipped with an **NVIDIA RTX 4070 GPU** with **chunk_size: 6** enabled.

### Real-World Metrics:
- **Time to First Byte (TTFB)**: **~380ms** (From request sent to first audio data arrival)
- **Real-Time Factor (RTF)**: **0.7 - 0.9** (Generating 1s of audio in 0.7s - 0.9s)
- **Throughput**: Zero-overhead binary delivery suitable for production-scale interaction.

---

## Features

* **Sub-500ms Latency**: Industry-leading response time for real-time voice interaction.
* **X-Vector Only Mode**: Support for pure speaker embedding cloning, which **100% eliminates prompt leakage** (no more hallucinating or repeating reference text).
* **Configurable Streaming Buffer**: Prevent audio stuttering during network fluctuations by buffering tokens before sending (`chunk_size`).
* **Server-Side Pre-Buffering**: Accumulate initial chunks before sending the first packet (`pre_buffer`) to provide a stable initial stream.
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
  --ref-audio reference.wav \
  --port 9000 \
  --chunk-size 6 \
  --pre-buffer 2
```

---

## API Reference

### Streaming TTS (Binary)
**Endpoint**: `POST /tts/stream`

**Request Body**:
```json
{
  "text": "Hello world.",
  "language": "English",
  "temperature": 0.5,
  "client_id": "user_123"
}
```

**Response**: `audio/l16;rate=24000`
- The server returns a continuous stream of raw **PCM 16-bit (Little Endian), Mono, 24,000Hz** bytes.
- There is no JSON wrapping or Base64 encoding. Every byte is actual audio data.

---

## Configuration

| CLI Argument | Environment Variable | Default | Description |
|--------------|----------------------|---------|-------------|
| `--model-path` | `MODEL_PATH` | `./Qwen3-TTS-12Hz-1.7B-Base` | Path to model weights |
| `--ref-audio` | `REF_AUDIO_PATH` | `None` | Path to reference audio for cloning (Required) |
| `--host` | `HOST` | `0.0.0.0` | Server host |
| `--port` | `PORT` | `9000` | Server port |
| `--chunk-size` | *None* | `1` | Global tokens to buffer before decoding/sending (Higher = better RTF) |
| `--pre-buffer` | *None* | `0` | Number of chunks to buffer on server before sending the first packet |
| `client_id` | *API Only* | `"default"` | Unique ID per user to enable parallel processing |

---

## WebUI Testing

A ready-to-use HTML client (`webui.html`) is included to test the streaming API.

1. Start the server (e.g., on port 9000).
2. Open `webui.html` in any modern web browser.
3. Configure your **Server URL** and **Client ID**.
4. Adjust **Language** and **Temperature** as needed.
5. Click **Play** to start synthesis. 
6. Observe the **Buffered** count in the stats board.

---

---

## Client Integration Guide (Binary)

### 1. High-Level Architecture
1.  **Request**: Send a `POST` request to `/tts/stream`.
2.  **Stream Consumption**: Use `fetch` and `response.body.getReader()`.
3.  **Conversion**: Convert the incoming `Uint8Array` (bytes) to `Int16Array`, then normalize to `Float32` for Web Audio.
4.  **Playback**: Schedule buffers using `AudioContext.createBufferSource()`.

### 2. JavaScript Implementation

```javascript
async function speakBinary(text) {
    const audioCtx = new AudioContext({ sampleRate: 24000 });
    let nextStartTime = audioCtx.currentTime;

    const response = await fetch('http://localhost:9000/tts/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language: "Chinese" })
    });

    const reader = response.body.getReader();

    while (true) {
        const { done, value } = await reader.read(); // value is Uint8Array
        if (done) break;

        // Convert PCM16 bytes to Float32
        const int16Array = new Int16Array(value.buffer, value.byteOffset, value.byteLength / 2);
        const float32Array = new Float32Array(int16Array.length);
        for (let i = 0; i < int16Array.length; i++) {
            float32Array[i] = int16Array[i] / 32768.0;
        }

        // Create and play buffer
        const buffer = audioCtx.createBuffer(1, float32Array.length, 24000);
        buffer.getChannelData(0).set(float32Array);
        
        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(audioCtx.destination);
        
        const startTime = Math.max(audioCtx.currentTime, nextStartTime);
        source.start(startTime);
        nextStartTime = startTime + buffer.duration;
    }
}
```

### 3. Key Optimization Tips
- **Binary Conversion**: Using `Int16Array` view on the `Uint8Array.buffer` is extremely fast and avoids manual byte shifting.
- **Sample Rate**: Ensure your `AudioContext` is locked to **24000Hz** to match the server output and avoid browser-side resampling.
- **No SSE Overhead**: Since there's no JSON parsing, you can process packets of any size immediately as they arrive.

---

## Credits

Developed based on the excellent work of the **Qwen Team**. This extension aims to provide the community with a high-speed serving alternative for real-time AI applications.

## License

Adheres to the original [License](LICENSE) provided by the Qwen team.
