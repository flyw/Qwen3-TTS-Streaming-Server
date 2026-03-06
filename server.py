import os
import io
import base64
import time
import logging
import json
import threading
import asyncio
import argparse
from typing import Optional, List, AsyncGenerator
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

# ================= Configuration & Logging =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Qwen3-TTS")

MODEL_PATH = os.getenv("MODEL_PATH", "./Qwen3-TTS-12Hz-1.7B-Base")
REF_AUDIO_PATH = os.getenv("REF_AUDIO_PATH", None)
REF_TEXT = os.getenv("REF_TEXT", None)
PORT = int(os.getenv("PORT", "9000"))
HOST = os.getenv("HOST", "0.0.0.0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

model_wrapper = None
default_voice_prompt = None
GLOBAL_SAVE_ENABLED = False 
inference_lock = None

app = FastAPI(title="Qwen3-TTS Precision Streaming Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class TTSRequest(BaseModel):
    text: str
    language: str = "Chinese"
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    max_new_tokens: int = 2048
    temperature: float = 0.5

def audio_to_base64_wav(wav: np.ndarray, sr: int) -> str:
    """Convert numpy array to standard WAV format encoded in Base64."""
    buffer = io.BytesIO()
    # Force PCM_16 to ensure stable byte count (2 bytes per sample)
    sf.write(buffer, wav, sr, format='WAV', subtype='PCM_16')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# ================= Precision Sliding Window Streamer =================
class AudioTokenStreamer(BaseStreamer):
    """
    Handles real-time audio token streaming with a sliding window approach
    to maintain audio continuity and minimize latency.
    """
    def __init__(self, model_wrapper, queue, loop, voice_prompt=None, save_enabled=False):
        self.model_wrapper = model_wrapper
        self.queue = queue
        self.loop = loop
        self.token_count = 0
        self.start_time = None
        
        # Get dynamic upsample rate (typically 2000 for 12Hz models)
        self.upsample_rate = model_wrapper.model.speech_tokenizer.get_decode_upsample_rate()
        
        # Reference audio context
        self.ref_code = None
        if voice_prompt and len(voice_prompt) > 0:
            self.ref_code = voice_prompt[0].ref_code.cpu().view(-1, 16)
            
        self.all_tokens_history = [] 
        self.context_window = 12 # Maintain 1-second sliding window
        self.last_total_samples = 0 # Global counter for decoded samples
        
        self.save_enabled = save_enabled
        self.final_audio_segments = []
        
        t_cfg = model_wrapper.model.config.talker_config
        self.special_token_ids = {t_cfg.codec_bos_id, t_cfg.codec_eos_token_id, t_cfg.codec_pad_id}

    def handle_forward_token(self, codec_ids):
        if self.start_time is None:
            self.start_time = time.time()

        token_id = int(codec_ids.view(-1)[0])
        if token_id in self.special_token_ids:
            return

        self.token_count += 1
        new_token = codec_ids.cpu().view(1, 16)
        self.all_tokens_history.append(new_token)
        
        try:
            with torch.no_grad():
                # 1. Construct sliding window input
                history_segment = self.all_tokens_history[-(self.context_window + 1):]
                
                if len(self.all_tokens_history) <= self.context_window and self.ref_code is not None:
                    needed_ref = self.context_window - len(self.all_tokens_history) + 1
                    ref_segment = self.ref_code[-needed_ref:]
                    decode_input = torch.cat([ref_segment] + history_segment, dim=0)
                else:
                    decode_input = torch.cat(history_segment, dim=0)

                # 2. Decode the current window
                wavs, sr = self.model_wrapper.model.speech_tokenizer.decode([{"audio_codes": decode_input}])
                window_wav = wavs[0]
                
                # 3. Precisely calculate incremental samples
                # Absolute global sample position at the end of current token
                current_total_samples = self.token_count * self.upsample_rate
                # Number of samples to extract for this chunk
                num_to_extract = current_total_samples - self.last_total_samples
                
                # Precise slicing from the end of the window
                new_audio_chunk = window_wav[-num_to_extract:]
                self.last_total_samples = current_total_samples
                
                # 4. Edge smoothing
                if self.token_count == 1:
                    # Apply fade-in to the first 400 samples to eliminate initial pop noise
                    new_audio_chunk[:400] *= np.linspace(0, 1, 400)

                if self.save_enabled:
                    self.final_audio_segments.append(new_audio_chunk)
                    
                audio_b64 = audio_to_base64_wav(new_audio_chunk, sr)
            
            self.loop.call_soon_threadsafe(
                self.queue.put_nowait, 
                {"type": "audio", "data": audio_b64, "index": self.token_count}
            )
        except Exception as e:
            logger.error(f"Precision decode error: {e}")

    def put(self, value): pass
    def end(self):
        if self.save_enabled and self.final_audio_segments:
            try:
                full_audio = np.concatenate(self.final_audio_segments)
                filename = f"precision_stream_{int(time.time())}.wav"
                sf.write(filename, full_audio, 24000)
                print(f"DEBUG: [Audio saved successfully] -> {filename}")
            except Exception as e:
                logger.error(f"Failed to save audio: {e}")
        self.loop.call_soon_threadsafe(self.queue.put_nowait, {"type": "done"})

# ================= Server Logic =================
@app.on_event("startup")
def load_model():
    global model_wrapper, default_voice_prompt
    logger.info(f"Loading Qwen3-TTS...")
    model_wrapper = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=DEVICE, dtype=DTYPE)
    
    # Monkey patch to bypass internal validation if necessary
    if hasattr(model_wrapper.model.talker, "_validate_model_kwargs"):
        model_wrapper.model.talker._validate_model_kwargs = lambda *args, **kwargs: None
        
    if REF_AUDIO_PATH and os.path.exists(REF_AUDIO_PATH):
        logger.info(f"Loading default voice prompt from {REF_AUDIO_PATH}")
        default_voice_prompt = model_wrapper.create_voice_clone_prompt(ref_audio=REF_AUDIO_PATH, ref_text=REF_TEXT)
    else:
        logger.warning("No reference audio provided. System will require ref_audio/ref_text in request or use default behavior.")
    logger.info("Server is Ready.")

async def generate_token_stream(request: TTSRequest) -> AsyncGenerator[str, None]:
    global inference_lock
    if inference_lock is None:
        inference_lock = asyncio.Lock()
    
    async with inference_lock:
        # Prepend space and punctuation to mitigate ICL model hallucinations at the beginning
        clean_text = " 。" + request.text.strip()
        
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        streamer = AudioTokenStreamer(model_wrapper, queue, loop, voice_prompt=default_voice_prompt, save_enabled=GLOBAL_SAVE_ENABLED)

        def forward_hook(module, input, output):
            if hasattr(output, "hidden_states") and isinstance(output.hidden_states, tuple):
                if len(output.hidden_states) > 1:
                    codec_ids = output.hidden_states[1] 
                    if codec_ids is not None and (codec_ids.dim() < 3 or codec_ids.shape[1] == 1):
                        streamer.handle_forward_token(codec_ids)

        # Robust Monkey Patch to prevent duplicate streamer passing
        original_talker_generate = model_wrapper.model.talker.generate
        def patched_generate(*args, **kwargs):
            kwargs.pop("streamer", None)
            return original_talker_generate(*args, streamer=streamer, **kwargs)
        
        model_wrapper.model.talker.generate = patched_generate
        hook_handle = model_wrapper.model.talker.register_forward_hook(forward_hook)

        def run_inference():
            try:
                model_wrapper.generate_voice_clone(
                    text=clean_text, 
                    voice_clone_prompt=default_voice_prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature
                )
            except Exception as e:
                logger.error(f"Inference Thread Error: {e}")
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
            finally:
                # Remove hook and restore generate method before signaling completion
                hook_handle.remove()
                model_wrapper.model.talker.generate = original_talker_generate
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})

        threading.Thread(target=run_inference).start()
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        
        while True:
            item = await queue.get()
            if item["type"] == "done":
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"

@app.post("/tts/stream")
async def tts_stream(request: TTSRequest):
    logger.info(f"Received request: {request.json()}")
    return StreamingResponse(generate_token_stream(request), media_type="text/event-stream")

@app.get("/health")
async def health(): return {"status": "ready"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS Precision Streaming Server")
    parser.add_argument("--save", action="store_true", help="Enable global saving of audio segments")
    parser.add_argument("--ref-audio-file", type=str, help="Path to reference audio file")
    parser.add_argument("--ref-text-file", type=str, help="Path to reference text file")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to the model directory")
    parser.add_argument("--host", type=str, default=HOST, help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=PORT, help="Port to bind the server to")
    
    args = parser.parse_args()
    
    GLOBAL_SAVE_ENABLED = args.save
    if args.ref_audio_file:
        REF_AUDIO_PATH = args.ref_audio_file
        logger.info(f"Using reference audio from: {REF_AUDIO_PATH}")
        
    if args.ref_text_file:
        if os.path.exists(args.ref_text_file):
            with open(args.ref_text_file, "r", encoding="utf-8") as f:
                REF_TEXT = f.read().strip()
            logger.info(f"Using reference text from file: {args.ref_text_file}")
        else:
            logger.error(f"Reference text file not found: {args.ref_text_file}")
            
    MODEL_PATH = args.model_path
    HOST = args.host
    PORT = args.port
    
    uvicorn.run(app, host=HOST, port=PORT)
