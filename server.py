import os
import io
import base64
import time
import logging
import json
import threading
import asyncio
import argparse
from typing import Optional, List, AsyncGenerator, Dict
import tempfile

import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from transformers.generation.streamers import BaseStreamer

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

import contextvars
from collections import defaultdict

# ================= Configuration & Logging =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Qwen3-TTS")

# [Optimization 1]: Enable TF32 Hardware Acceleration
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Storage for per-client locks to ensure serial processing for each user while allowing cross-user parallelism
client_locks = defaultdict(asyncio.Lock)
# Track the stop_event of the active task for each client to allow proactive interruption
active_stop_events: Dict[str, threading.Event] = {}
# Counter to track interrupt events for flushing the queue
interrupt_counters = defaultdict(int)

# Context variable to route tokens to the correct request-specific streamer in a thread-safe way
active_streamer: contextvars.ContextVar[Optional[BaseStreamer]] = contextvars.ContextVar("active_streamer", default=None)

MODEL_PATH = os.getenv("MODEL_PATH", "./Qwen3-TTS-12Hz-1.7B-Base")
REF_AUDIO_PATH = os.getenv("REF_AUDIO_PATH", None)
PORT = int(os.getenv("PORT", "9000"))
HOST = os.getenv("HOST", "0.0.0.0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# [Optimization 2]: Dynamic DTYPE Selection
if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
    DTYPE = torch.bfloat16
    logger.info("Performance mode: Using Bfloat16")
else:
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
    logger.info(f"Performance mode: Using {DTYPE}")

model_wrapper = None
default_voice_prompt = None
GLOBAL_SAVE_ENABLED = False 
GLOBAL_CHUNK_SIZE = 1
GLOBAL_PRE_BUFFER = 0

app = FastAPI(title="Qwen3-TTS Thread-Safe Parallel Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class TTSRequest(BaseModel):
    text: str
    language: str = "Chinese"
    ref_audio: Optional[str] = None
    max_new_tokens: int = 2048
    temperature: float = 0.5
    client_id: Optional[str] = Field(default="default", description="Unique ID for each client to maintain sequence")

def get_pcm_bytes(wav: np.ndarray) -> bytes:
    """Convert float32 to int16 bytes for raw binary transfer."""
    wav = np.clip(wav.flatten(), -1.0, 1.0)
    return (wav * 32767).astype(np.int16).tobytes()

# ================= Precision Sliding Window Streamer =================
class AudioTokenStreamer(BaseStreamer):
    """
    Handles real-time audio token streaming with a sliding window approach
    to maintain audio continuity and minimize latency.
    """
    def __init__(self, model_wrapper, queue, loop, voice_prompt=None, save_enabled=False, chunk_size=1, pre_buffer=0, stop_event=None):
        self.model_wrapper = model_wrapper
        self.queue = queue
        self.loop = loop
        self.token_count = 0
        self.start_time = None
        self.stop_event = stop_event
        
        # [Optimization]: Batch decoding buffer
        self.chunk_size = max(1, chunk_size)
        self.pending_tokens = [] 
        
        self.skip_tokens = 0 
        self.actual_sent_count = 0
        self.pre_buffer_limit = pre_buffer
        self.pre_buffer_storage = [] # Stores pcm bytes
        
        self.upsample_rate = model_wrapper.model.speech_tokenizer.get_decode_upsample_rate()
        self.sample_rate = 24000
        
        self.ref_code = None
        if voice_prompt and len(voice_prompt) > 0 and voice_prompt[0].ref_code is not None:
            # Keep on GPU if possible for faster concatenation
            self.ref_code = voice_prompt[0].ref_code.view(-1, 16)
            
        self.all_tokens_history = [] 
        self.context_window = 12
        self.last_total_samples = 0
        
        self.save_enabled = save_enabled
        self.final_audio_segments = []
        
        t_cfg = model_wrapper.model.config.talker_config
        self.special_token_ids = {t_cfg.codec_bos_id, t_cfg.codec_eos_token_id, t_cfg.codec_pad_id}

    def _send_audio_packet(self, wav_chunk):
        """Send raw PCM bytes to queue."""
        if wav_chunk is None or len(wav_chunk) == 0: return
        pcm_bytes = get_pcm_bytes(wav_chunk)
        try:
            if self.actual_sent_count < self.pre_buffer_limit:
                self.pre_buffer_storage.append(pcm_bytes)
                self.actual_sent_count += 1
                if self.actual_sent_count == self.pre_buffer_limit:
                    all_pre_bytes = b"".join(self.pre_buffer_storage)
                    self.loop.call_soon_threadsafe(self.queue.put_nowait, all_pre_bytes)
                    self.pre_buffer_storage = []
                return

            self.loop.call_soon_threadsafe(self.queue.put_nowait, pcm_bytes)
        except Exception as e:
            logger.error(f"Error sending pcm packet: {e}")

    def _decode_batch(self):
        """The core optimization: Decode multiple tokens in a single model forward pass."""
        if not self.pending_tokens: return
        
        num_new_tokens = len(self.pending_tokens)
        self.pending_tokens = []
        effective_token_count = self.token_count - self.skip_tokens
        
        try:
            with torch.no_grad():
                # 1. Sliding window with Batch Support
                history_segment = self.all_tokens_history[-(self.context_window + num_new_tokens):]
                
                if len(self.all_tokens_history) <= self.context_window and self.ref_code is not None:
                    needed_ref = self.context_window - len(self.all_tokens_history) + num_new_tokens
                    ref_segment = self.ref_code[-needed_ref:]
                    decode_input = torch.cat([ref_segment] + history_segment, dim=0)
                else:
                    decode_input = torch.cat(history_segment, dim=0)

                # 2. Single Decode call for multiple tokens
                wavs, sr = self.model_wrapper.model.speech_tokenizer.decode([{"audio_codes": decode_input}])
                window_wav = wavs[0]
                self.sample_rate = sr
                
                # 3. Precisely extract audio for the entire batch
                current_total_samples = effective_token_count * self.upsample_rate
                num_to_extract = current_total_samples - self.last_total_samples
                new_audio_chunk = window_wav[-num_to_extract:]
                self.last_total_samples = current_total_samples
                
                # 4. First packet fade-in
                if effective_token_count <= num_new_tokens:
                    fade_len = min(1000, len(new_audio_chunk))
                    new_audio_chunk[:fade_len] *= np.linspace(0, 1, fade_len)

                if self.save_enabled:
                    self.final_audio_segments.append(new_audio_chunk)
                
                self._send_audio_packet(new_audio_chunk)
                    
        except Exception as e:
            logger.error(f"Batch decode error: {e}")

    def handle_forward_token(self, codec_ids):
        # [INTERRUPTION]: Raise error to stop generate() immediately if client disconnected
        if self.stop_event and self.stop_event.is_set():
            raise InterruptedError("Client disconnected, stopping inference.")

        if self.start_time is None:
            self.start_time = time.time()

        token_id = int(codec_ids.view(-1)[0])
        if token_id in self.special_token_ids: return

        self.token_count += 1
        if self.token_count <= self.skip_tokens: return

        # 1. Accumulate tokens
        new_token = codec_ids.view(1, 16)
        self.all_tokens_history.append(new_token)
        self.pending_tokens.append(new_token)
        
        # 2. Trigger decode: either hit chunk_size, or it's the very first token (to keep TTFT low)
        if len(self.pending_tokens) >= self.chunk_size or (self.token_count - self.skip_tokens) == 1:
            self._decode_batch()

    def put(self, value): pass
    def end(self):
        # Flush remaining tokens
        self._decode_batch()
        
        if self.pre_buffer_storage:
            try:
                all_pre_bytes = b"".join(self.pre_buffer_storage)
                self.loop.call_soon_threadsafe(self.queue.put_nowait, all_pre_bytes)
                self.pre_buffer_storage = []
            except: pass

        if self.save_enabled and self.final_audio_segments:
            try:
                full_audio = np.concatenate(self.final_audio_segments)
                filename = f"precision_stream_{int(time.time())}.wav"
                sf.write(filename, full_audio, self.sample_rate)
            except: pass
        # Use None to signal end of binary stream
        self.loop.call_soon_threadsafe(self.queue.put_nowait, None)

# ================= Server Logic =================
def global_forward_hook(module, input, output):
    """Global hook that routes codec_ids to the streamer in the current thread context."""
    if hasattr(output, "hidden_states") and isinstance(output.hidden_states, tuple):
        if len(output.hidden_states) > 1:
            codec_ids = output.hidden_states[1] 
            if codec_ids is not None and (codec_ids.dim() < 3 or codec_ids.shape[1] == 1):
                streamer = active_streamer.get()
                if streamer and hasattr(streamer, "handle_forward_token"):
                    streamer.handle_forward_token(codec_ids)

@app.on_event("startup")
def load_model():
    global model_wrapper, default_voice_prompt
    logger.info(f"Loading Qwen3-TTS and preparing thread-safe environment...")
    
    # Global gradient disabling
    torch.set_grad_enabled(False)
    
    model_wrapper = Qwen3TTSModel.from_pretrained(
        MODEL_PATH, 
        device_map=DEVICE, 
        dtype=DTYPE
    )
    
    # [Optimization 3]: Selective Torch Compile
    # We compile the speech_tokenizer (VQ decoder) which is compute-heavy 
    # but keep the talker uncompiled to ensure Forward Hooks work correctly.
    if hasattr(torch, 'compile') and DEVICE == "cuda":
        logger.info("Compiling speech_tokenizer for faster audio decoding...")
        try:
            model_wrapper.model.speech_tokenizer.model = torch.compile(
                model_wrapper.model.speech_tokenizer.model, 
                mode="reduce-overhead"
            )
        except Exception as e:
            logger.warning(f"torch.compile for tokenizer failed: {e}")

    # 1. Global Monkey Patch: inject the context-aware streamer into the generate call
    original_talker_generate = model_wrapper.model.talker.generate
    def thread_safe_generate(*args, **kwargs):
        # Retrieve the streamer assigned to the current thread/context
        s = active_streamer.get()
        kwargs.pop("streamer", None)
        return original_talker_generate(*args, streamer=s, **kwargs)
    
    model_wrapper.model.talker.generate = thread_safe_generate
    
    # 2. Global Forward Hook: register once, used by all parallel requests
    model_wrapper.model.talker.register_forward_hook(global_forward_hook)
    
    if hasattr(model_wrapper.model.talker, "_validate_model_kwargs"):
        model_wrapper.model.talker._validate_model_kwargs = lambda *args, **kwargs: None
        
    if REF_AUDIO_PATH and os.path.exists(REF_AUDIO_PATH):
        logger.info(f"Loading default voice prompt in X-VECTOR MODE")
        default_voice_prompt = model_wrapper.create_voice_clone_prompt(
            ref_audio=REF_AUDIO_PATH,
            ref_text=None,
            x_vector_only_mode=True
        )

    # [Optimization 4]: Pre-warmup
    if DEVICE == "cuda":
        logger.info("Warming up CUDA kernels...")
        with torch.inference_mode():
            model_wrapper.generate_voice_clone(
                text="Warmup.", 
                voice_clone_prompt=default_voice_prompt, 
                max_new_tokens=24
            )
        logger.info("Warmup complete.")

    logger.info("Server is Parallel-Ready.")

async def generate_token_stream(request: TTSRequest) -> AsyncGenerator[bytes, None]:
    # Record the interrupt version at the moment the request arrived
    my_interrupt_version = interrupt_counters[request.client_id]

    # Acquire lock specific to this client_id to maintain sequence for each user
    # Multiple sentences from the same LLM stream will queue up here.
    async with client_locks[request.client_id]:
        # [NEW]: Check if an interrupt happened while we were waiting in the queue
        if interrupt_counters[request.client_id] > my_interrupt_version:
            logger.info(f"FLUSHING queued request for ClientID: {request.client_id} due to previous interrupt call.")
            return

        # Standard space padding
        clean_text = " " + request.text.strip()
        
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        stop_event = threading.Event()
        
        # Register the new stop_event as the active one for this client
        active_stop_events[request.client_id] = stop_event
        
        streamer = AudioTokenStreamer(
            model_wrapper, 
            queue, 
            loop, 
            voice_prompt=default_voice_prompt, 
            save_enabled=GLOBAL_SAVE_ENABLED,
            chunk_size=GLOBAL_CHUNK_SIZE,
            pre_buffer=GLOBAL_PRE_BUFFER,
            stop_event=stop_event
        )

        def run_inference():
            # Set the context for this specific thread
            token = active_streamer.set(streamer)
            inf_start = time.time()
            try:
                model_wrapper.generate_voice_clone(
                    text=clean_text, 
                    voice_clone_prompt=default_voice_prompt,
                    language=request.language,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature
                )
                inf_end = time.time()
                inf_dur = inf_end - inf_start
                
                # Calculate precise audio duration from token count
                audio_dur = (streamer.token_count * streamer.upsample_rate) / streamer.sample_rate
                rtf = inf_dur / audio_dur if audio_dur > 0 else 0
                
                logger.info("-" * 30)
                logger.info(f"PROCESSED: \"{clean_text.strip()[:30]}...\"")
                logger.info(f"METRICS:   Inference: {inf_dur:.2f}s | Audio: {audio_dur:.2f}s | RTF: {rtf:.4f}")
                logger.info("-" * 30)
                
            except InterruptedError:
                logger.info(f"Interrupted: Request cancelled/preempted (ClientID: {request.client_id})")
            except Exception as e:
                logger.error(f"Inference Thread Error: {e}")
            finally:
                active_streamer.reset(token)
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # Run in a separate thread but maintain context propagation
        inf_task = asyncio.create_task(asyncio.to_thread(run_inference))
        
        try:
            while True:
                item = await queue.get()
                if item is None: break
                yield item
        finally:
            # Signal the thread to stop and wait for it to finish before releasing lock
            stop_event.set()
            # Clean up the global tracker only if it's still OUR event
            if active_stop_events.get(request.client_id) == stop_event:
                active_stop_events.pop(request.client_id, None)
            await inf_task

@app.post("/tts/stream")
async def tts_stream(request: TTSRequest):
    logger.info(f"Received request (ClientID: {request.client_id}): {request.text[:20]}...")
    return StreamingResponse(generate_token_stream(request), media_type="audio/l16;rate=24000")

@app.post("/tts/interrupt")
async def interrupt_client(client_id: str = "default"):
    """Explicitly interrupt the active task and clear all queued tasks for a specific client."""
    # Increment counter to cause all currently waiting requests to exit as soon as they get the lock
    interrupt_counters[client_id] += 1
    
    if client_id in active_stop_events:
        logger.info(f"Manual interruption requested for ClientID: {client_id}. All queued tasks will be flushed.")
        active_stop_events[client_id].set()
        return {"status": "interrupted", "client_id": client_id, "queue": "flushed"}
    return {"status": "idle", "client_id": client_id, "queue": "flushed"}

@app.get("/health")
async def health(): return {"status": "ready"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS Precision Streaming Server")
    parser.add_argument("--save", action="store_true", help="Enable global saving of audio segments")
    parser.add_argument("--ref-audio", type=str, required=True, help="Path to reference audio file (required)")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to the model directory")
    parser.add_argument("--host", type=str, default=HOST, help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=PORT, help="Port to bind the server to")
    parser.add_argument("--chunk-size", type=int, default=1, help="Default tokens to buffer before sending")
    parser.add_argument("--pre-buffer", type=int, default=0, help="Number of chunks to buffer on server before sending")

    args = parser.parse_args()

    GLOBAL_SAVE_ENABLED = args.save

    if not os.path.exists(args.ref_audio):
        logger.error(f"Reference audio file not found: {args.ref_audio}")
        logger.error("Please provide a valid reference audio file using --ref-audio")
        exit(1)

    REF_AUDIO_PATH = args.ref_audio
    logger.info(f"Using reference audio from: {REF_AUDIO_PATH}")
            
    MODEL_PATH = args.model_path
    HOST = args.host
    PORT = args.port
    GLOBAL_CHUNK_SIZE = args.chunk_size
    GLOBAL_PRE_BUFFER = args.pre_buffer
    
    uvicorn.run(app, host=HOST, port=PORT)
