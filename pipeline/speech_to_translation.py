"""
pipeline/speech_to_translation.py  —  web-ready version

Key change from desktop version:
  process_audio_bytes() accepts a pre-saved audio path (from Gradio)
  instead of recording from mic. Everything else is identical.
  process_audio() (mic version) is kept for CLI use.
"""

import os
import threading
import numpy as np
import soundfile as sf

from stt.whisper_engine import transcribe_audio
from translation.translator import translate_text
from tts.mms_tts import speak_preserved, speak, generate_audio
from tts.language_manager import WHISPER_TO_NLLB

LANGUAGE_MENU = {
    "1": ("eng_Latn", "English"),
    "2": ("hin_Deva", "Hindi"),
    "3": ("urd_Arab", "Urdu"),
    "4": ("spa_Latn", "Spanish"),
    "5": ("fra_Latn", "French"),
    "6": ("deu_Latn", "German"),
    "7": ("por_Latn", "Portuguese"),
    "8": ("ara_Arab", "Arabic"),
}

_session_voice_profile = None
_profile_lock = threading.Lock()


def reset_voice_profile():
    global _session_voice_profile
    with _profile_lock:
        _session_voice_profile = None


def _build_profile_bg(audio_path: str):
    global _session_voice_profile
    try:
        from speech_preservation.voice_converter import build_voice_profile
        profile = build_voice_profile(audio_path)
        with _profile_lock:
            if _session_voice_profile is None:
                _session_voice_profile = profile
    except Exception as e:
        print(f"[Voice profile] {e}")


def process_audio_bytes(
    manager,
    audio_path: str,            # path to pre-saved wav (from Gradio)
    source_lang,
    target_lang: str,
    preserve_voice: bool = True,
    vc_strength: float = 0.6,
) -> tuple:
    """
    Web pipeline: audio_path → STT → translate → TTS → output_path.
    Returns (source_text, translated_text, detected_language, output_audio_path).
    """
    global _session_voice_profile

    # ── Build voice profile in background while STT runs ─────────────
    profile_thread = None
    if preserve_voice:
        with _profile_lock:
            need_profile = _session_voice_profile is None
        if need_profile:
            profile_thread = threading.Thread(
                target=_build_profile_bg, args=(audio_path,), daemon=True
            )
            profile_thread.start()

    # ── STT ───────────────────────────────────────────────────────────
    whisper_lang = None
    if source_lang is not None:
        rev = {v: k for k, v in WHISPER_TO_NLLB.items()}
        whisper_lang = rev.get(source_lang)

    stt_result        = transcribe_audio(manager.whisper_model, audio_path, language=whisper_lang)
    source_text       = stt_result["text"]
    detected_language = stt_result["language"]

    if source_lang is None:
        source_lang = WHISPER_TO_NLLB.get(detected_language, "eng_Latn")

    # ── Wait for profile ──────────────────────────────────────────────
    if profile_thread is not None:
        profile_thread.join(timeout=5.0)

    # ── Translation ───────────────────────────────────────────────────
    translated_text = (
        source_text if source_lang == target_lang
        else translate_text(
            text=source_text,
            tokenizer=manager.tokenizer,
            translator_model=manager.translator_model,
            source_lang=source_lang,
            target_lang=target_lang,
        )
    )

    # ── TTS → save to file (web needs a file path, not playback) ──────
    output_path = _synthesise_to_file(
        text=translated_text,
        language_code=target_lang,
        preserve_voice=preserve_voice,
        vc_strength=vc_strength,
    )

    return source_text, translated_text, detected_language, output_path


def _synthesise_to_file(
    text: str,
    language_code: str,
    preserve_voice: bool,
    vc_strength: float,
) -> str:
    """Generate TTS audio and return file path for Gradio to serve."""
    import hashlib, os
    from tts.mms_tts import generate_audio, _cache_path

    out_path = _cache_path(text, language_code, suffix=f"_vc{vc_strength:.1f}" if preserve_voice else "")

    if os.path.exists(out_path):
        return out_path

    audio, sr = generate_audio(text, language_code)

    if preserve_voice:
        try:
            from speech_preservation.voice_converter import apply_voice_conversion
            with _profile_lock:
                profile = _session_voice_profile
            if profile:
                audio = apply_voice_conversion(audio, sr, profile, vc_strength)
        except Exception as e:
            print(f"[VC] {e} — using plain TTS")

    sf.write(out_path, audio, sr)
    return out_path


# ── CLI version (unchanged) ───────────────────────────────────────────────

def process_audio(
    manager,
    source_lang,
    target_lang: str,
    status_callback=None,
    preserve_voice: bool = True,
    vc_strength: float = 0.6,
) -> tuple:
    from audio.recorder import record_audio
    audio_path = record_audio(status_callback=status_callback)
    src, trl, det, out = process_audio_bytes(
        manager, audio_path, source_lang, target_lang, preserve_voice, vc_strength
    )
    # Play locally for CLI
    import sounddevice as sd
    data, sr = sf.read(out, dtype="float32")
    sd.play(data, sr)
    sd.wait()
    return src, trl, det
