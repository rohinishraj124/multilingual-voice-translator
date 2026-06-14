import os
import time
import uuid
import subprocess
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from models.model_manager import ModelManager
from pipeline.speech_to_translation import process_audio_bytes, LANGUAGE_MENU, reset_voice_profile

# ── Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Voice Translator API")

# Ensure directories exist
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs/tts_cache"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load models
print("\nLoading models…")
manager = ModelManager()
manager.load_models()
print("All models ready.\n")

# Mapping for NLLB
NLLB_BY_NAME = {name: code for code, name in LANGUAGE_MENU.values()}

# ── API Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/translate")
async def translate_endpoint(
    audio: UploadFile = File(...),
    target_lang: str = Form("Hindi"),
    preserve_voice: bool = Form(True),
    vc_strength: float = Form(0.6)
):
    # 1. Save uploaded file (likely .webm or .ogg from browser)
    session_id = str(uuid.uuid4())
    webm_path = os.path.join(UPLOAD_DIR, f"{session_id}.webm")
    wav_path = os.path.join(UPLOAD_DIR, f"{session_id}.wav")

    with open(webm_path, "wb") as f:
        f.write(await audio.read())

    try:
        # 2. Convert to 16kHz WAV using FFmpeg
        # Browser audio often needs conversion for Whisper
        cmd = [
            "ffmpeg", "-y", "-i", webm_path,
            "-ar", "16000", "-ac", "1", wav_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # 3. Process through pipeline
        target_lang_code = NLLB_BY_NAME.get(target_lang, "hin_Deva")
        
        # Reset voice profile for new sessions if needed, 
        # or keep it if you want to remember the voice during the browser session.
        # For simplicity, we'll keep the session profile logic in process_audio_bytes.

        source_text, translated_text, detected, output_audio_path = process_audio_bytes(
            manager=manager,
            audio_path=wav_path,
            source_lang=None, # Auto-detect
            target_lang=target_lang_code,
            preserve_voice=preserve_voice,
            vc_strength=vc_strength,
        )

        # 4. Return results
        # output_audio_path is relative to project root, e.g., "outputs/tts_cache/xyz.wav"
        # We need a URL for the frontend.
        audio_url = f"/audio/{os.path.basename(output_audio_path)}"

        return {
            "source_text": source_text,
            "translated_text": translated_text,
            "detected_lang": detected,
            "audio_url": audio_url
        }

    except Exception as e:
        print(f"Error in translation: {e}")
        return {"error": str(e)}
    finally:
        # Cleanup input files
        if os.path.exists(webm_path): os.remove(webm_path)
        if os.path.exists(wav_path): os.remove(wav_path)

# ── Static Files ───────────────────────────────────────────────────────────

# Serve generated audio files
app.mount("/audio", StaticFiles(directory="outputs/tts_cache"), name="audio")

# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
