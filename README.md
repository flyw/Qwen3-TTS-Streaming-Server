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
- **Anti-Jitter Buffering**: Built-in server-side and client-side pre-buffering to ensure smooth playback even under high GPU load or network fluctuations.

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

### Streaming TTS (SSE)
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
*   **text**: The text to synthesize.
*   **language**: Target language (Chinese, English, Japanese, Korean, Cantonese).
*   **temperature**: Controls expressiveness (0.1 to 1.0).
*   **client_id**: Unique ID per user. Requests with the same ID are processed **sequentially**, while different IDs are processed **in parallel**.

**Response**: A `text/event-stream` returning JSON chunks:
- `{"type": "start", "total_chunks": 1}`: Signals the start of generation.
- `{"type": "audio", "data": "BASE64_WAV_CHUNK", "index": 1}`: Audio data chunk.
- `{"type": "done"}`: Signals completion.

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

## Client Integration Guide

### 1. High-Level Architecture
1.  **Request**: Send a `POST` request to `/tts/stream`.
2.  **Stream Consumption**: Use `fetch` and `ReadableStream`.
3.  **Decoding**: Decode Base64-encoded WAV chunks into `AudioBuffer`.
4.  **Scheduling**: Use the Web Audio API to schedule each chunk.

### 2. Complete JavaScript Implementation

```javascript
async function speak(text, options = {}) {
    const response = await fetch('http://localhost:9000/tts/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text,
            language: "Chinese",
            temperature: 0.5,
            ...options
        })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop();

        for (const line of lines) {
            if (line.startsWith("data: ")) {
                const jsonStr = line.replace("data: ", "").trim();
                const payload = JSON.parse(jsonStr);

                if (payload.type === "audio") {
                    // Convert Base64 to ArrayBuffer and play using Web Audio API
                    await playChunk(payload.data);
                }
            }
        }
    }
}
```

### 3. Key Optimization Tips

- **Next Start Time Tracking**: Never use `source.start(0)`. Always track `nextStartTime = startTime + buffer.duration` to ensure the next chunk starts exactly where the previous one ended.
- **Handling Network Jitter**: 
    - **Server Side**: Set `pre_buffer` (e.g., 3) to send a larger first packet.
    - **Client Side**: If your RTF is close to 1.0, wait for 2-3 chunks to arrive before starting the very first `source.start()`.
- **Sample Rate**: Qwen3-TTS outputs **24,000Hz**. Hardcoding this in your `AudioContext` prevents the browser from doing expensive resampling.
- **WAV Headers**: The server sends each chunk as a standard WAV file. Most modern browsers' `decodeAudioData` handle this seamlessly, even if headers are repeated per chunk.

---

## Credits

Developed based on the excellent work of the **Qwen Team**. This extension aims to provide the community with a high-speed serving alternative for real-time AI applications.

## License

Adheres to the original [License](LICENSE) provided by the Qwen team.
