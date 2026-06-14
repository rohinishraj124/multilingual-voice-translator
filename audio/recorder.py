"""
audio/recorder.py

FIX 1: Removed hardcoded DEVICE_ID=5 — this caused recording to silently fail
        or crash on machines where device 5 doesn't exist or isn't a microphone.
        Now auto-selects the system default input device.

FIX 2: Added voice-activity detection (VAD) so recording stops automatically
        after silence instead of always recording a fixed 5-second chunk.
        This makes live translation feel genuinely "live".

IMPROVEMENT: Accepts optional device_id and duration overrides via args.
"""

import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

OUTPUT_FILE = "outputs/input.wav"
SAMPLE_RATE = 16000       # 16 kHz — Whisper's native rate; avoids librosa resample
DURATION = 7              # fallback max duration (seconds)
SILENCE_THRESHOLD = 500   # RMS below this = silence
SILENCE_DURATION = 1.5    # seconds of silence before auto-stop


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


def record_audio(
    status_callback=None,
    device_id=None,       # None = system default mic
    max_duration=DURATION
) -> str:
    """
    Record audio from the microphone with voice-activity detection.
    Returns path to saved .wav file.
    """
    for i in range(3, 0, -1):
        if status_callback:
            status_callback(f"Starting in {i}...")
        time.sleep(1)

    if status_callback:
        status_callback("🎙️ Recording... (speak now)")

    chunk_size = int(SAMPLE_RATE * 0.1)   # 100 ms chunks
    max_chunks = int(max_duration / 0.1)
    silence_chunks_needed = int(SILENCE_DURATION / 0.1)

    frames = []
    silence_count = 0
    speech_started = False

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=device_id,           # FIX 1: None = system default
        blocksize=chunk_size
    ) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
            frames.append(chunk.copy())

            if _rms(chunk) > SILENCE_THRESHOLD:
                speech_started = True
                silence_count = 0
            elif speech_started:
                silence_count += 1
                if silence_count >= silence_chunks_needed:
                    break   # FIX 2: stop after sustained silence

    recording = np.concatenate(frames, axis=0)
    write(OUTPUT_FILE, SAMPLE_RATE, recording)

    if status_callback:
        status_callback("✅ Recording complete")

    return OUTPUT_FILE
