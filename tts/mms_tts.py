"""
tts/mms_tts.py  —  MMS-TTS with optional voice conversion
"""

import os
import hashlib
import logging
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

from transformers import AutoTokenizer, VitsModel
from tts.language_manager import get_mms_language

log = logging.getLogger(__name__)

CACHE_DIR = os.path.join("outputs", "tts_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_CACHE: dict = {}


def load_model(nllb_language: str):
    mms_language = get_mms_language(nllb_language)
    if mms_language in MODEL_CACHE:
        return MODEL_CACHE[mms_language]
    model_name = f"facebook/mms-tts-{mms_language}"
    print(f"\nLoading MMS Model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name)
    model.eval()
    MODEL_CACHE[mms_language] = (tokenizer, model)
    return tokenizer, model


def generate_audio(text: str, language_code: str):
    tokenizer, model = load_model(language_code)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform
    audio = waveform.squeeze().cpu().numpy().astype(np.float32)
    return audio, model.config.sampling_rate


def _cache_path(text: str, lang: str, suffix: str = "") -> str:
    key = f"{lang}{suffix}{text}"
    return os.path.join(CACHE_DIR, hashlib.md5(key.encode()).hexdigest() + ".wav")


def _play(path: str) -> None:
    data, sr = sf.read(path, dtype="float32")
    sd.play(data, sr)
    sd.wait()


def speak(text: str, language_code: str) -> None:
    """Plain MMS-TTS, no voice conversion."""
    if not text.strip():
        return
    try:
        path = _cache_path(text, language_code)
        if not os.path.exists(path):
            audio, sr = generate_audio(text, language_code)
            sf.write(path, audio, sr)
        _play(path)
    except Exception as e:
        log.error("[MMS speak] %s", e)


def speak_preserved(
    text: str,
    language_code: str,
    voice_profile: dict | None,
    strength: float = 0.6,
) -> None:
    """MMS-TTS output with voice conversion applied (~1-3s on CPU)."""
    if not text.strip():
        return

    if voice_profile is None:
        speak(text, language_code)
        return

    try:
        from speech_preservation.voice_converter import apply_voice_conversion

        path = _cache_path(text, language_code, suffix=f"_vc{strength:.1f}")

        if os.path.exists(path):
            _play(path)
            return

        audio, sr = generate_audio(text, language_code)
        audio = apply_voice_conversion(audio, sr, voice_profile, strength)
        sf.write(path, audio, sr)
        _play(path)

    except Exception as e:
        log.error("[MMS+VC] %s — falling back to plain MMS.", e)
        speak(text, language_code)
