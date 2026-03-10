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
  "chunk_size": 5,
  "pre_buffer": 3,
  "client_id": "user_123"
}
```
*   **chunk_size**: Number of tokens to buffer before sending a single SSE packet. (e.g., 12 tokens $\approx$ 1s of audio for 12Hz models).
*   **pre_buffer**: Number of initial chunks to buffer on the server before sending the very first data packet. This creates a "head start" for the player to prevent stuttering under load.
*   **client_id**: Unique ID per user. Requests with the same ID are processed **sequentially**, while different IDs are processed **in parallel**.

**Response**: A `text/event-stream` returning JSON chunks:
- `{"type": "start"}`: Signals the start of generation.
- `{"type": "audio", "data": "BASE64_WAV_CHUNK", "index": 1, "is_prebuffered": true}`: Audio data chunk.
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
| `pre_buffer` | *API Only* | `0` | Initial chunks to buffer on server for the first packet |
| `client_id` | *API Only* | `"default"` | Unique ID per user to enable parallel processing |

---

## WebUI Testing

A ready-to-use HTML client (`webui.html`) is included to test the streaming API. It features a high-performance **Anti-Jitter Buffer** implementation.

1. Start the server (e.g., on port 9000).
2. Open `webui.html` in any modern web browser.
3. Configure your **Server URL** and **Client ID**.
4. **Optimization Settings**:
    - **Chunk Size**: Adjust how many tokens are in each network packet. Increase this (e.g., to 10-15) for better efficiency on high-latency networks.
    - **Pre-buffer**: Set the initial buffer depth. If you experience stuttering (RTF < 1.2), set this to 3 or 4.
5. Click **Play** to start synthesis. 
6. Observe the **Buffered** count in the stats board. If it drops to 0 during playback, increase your `Pre-buffer`.

---

---

## Client Integration Guide

Integrating a streaming TTS requires handling **Server-Sent Events (SSE)** and managing an **Audio Jitter Buffer** to ensure gapless playback.

### 1. High-Level Architecture
1.  **Request**: Send a `POST` request to `/tts/stream` with `stream: true`.
2.  **Stream Consumption**: Use `fetch` and `ReadableStream` (since standard `EventSource` doesn't support POST).
3.  **Decoding**: Decode Base64-encoded WAV chunks into `AudioBuffer`.
4.  **Scheduling**: Use the Web Audio API to schedule each chunk at a precise `startTime` to avoid clicks or gaps.

### 2. Complete JavaScript Implementation

```javascript
class Qwen3TTSPlayer {
    constructor(sampleRate = 24000) {
        this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
        this.nextStartTime = 0; // Tracks the end time of the last scheduled chunk
        this.isPlaying = false;
    }

    /**
     * Synthesize and play text
     * @param {string} text - Text to speak
     * @param {Object} options - API options (chunk_size, pre_buffer, etc.)
     */
    async speak(text, options = {}) {
        // Ensure AudioContext is active (browsers block audio until user interaction)
        if (this.audioCtx.state === 'suspended') await this.audioCtx.resume();
        
        // Reset timing for a new sentence
        this.nextStartTime = this.audioCtx.currentTime;
        this.isPlaying = true;

        try {
            const response = await fetch('http://localhost:9000/tts/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text,
                    language: "Chinese",
                    chunk_size: 5,
                    pre_buffer: 3,
                    stream: true,
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
                buffer = lines.pop(); // Keep incomplete JSON line

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const jsonStr = line.replace("data: ", "").trim();
                        const payload = JSON.parse(jsonStr);

                        if (payload.type === "audio") {
                            await this.schedulePlayback(payload.data);
                        } else if (payload.type === "done") {
                            console.log("Stream finished");
                        }
                    }
                }
            }
        } catch (err) {
            console.error("Playback Error:", err);
        } finally {
            this.isPlaying = false;
        }
    }

    /**
     * Decodes Base64 to AudioBuffer and schedules it in the Web Audio timeline
     */
    async schedulePlayback(base64Data) {
        // Convert Base64 to ArrayBuffer
        const binary = atob(base64Data);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        
        // Decode audio data
        const audioBuffer = await this.audioCtx.decodeAudioData(bytes.buffer);
        
        // Create BufferSource
        const source = this.audioCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioCtx.destination);

        // Calculate start time: 
        // We use either the current time or the end of the previous chunk, whichever is later.
        const now = this.audioCtx.currentTime;
        const startTime = Math.max(now, this.nextStartTime);
        
        source.start(startTime);

        // Update the timeline
        this.nextStartTime = startTime + audioBuffer.duration;
    }
}

// Usage:
// const player = new Qwen3TTSPlayer();
// player.speak("欢迎使用通义千问语音大模型。");
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
