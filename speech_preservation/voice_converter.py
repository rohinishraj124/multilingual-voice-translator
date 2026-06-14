"""
speech_preservation/voice_converter.py

CPU-friendly voice conversion: builds a voice profile from the source
recording, then applies pitch shift + time stretch + spectral EQ to
MMS-TTS output so it sounds more like the original speaker.

Runs in ~1-3 seconds on an 8 GB CPU laptop.

Install:  pip install praat-parselmouth librosa
Optional: pip install pyrubberband  (better time-stretch quality)
          + rubberband CLI:
            Windows → breakfastquay.com/rubberband/
            Linux   → sudo apt install rubberband-cli
            macOS   → brew install rubberband
"""

from __future__ import annotations
import logging
import numpy as np

log = logging.getLogger(__name__)

# MMS-TTS outputs at ~150 Hz neutral pitch
MMS_BASE_PITCH = 150.0
BLEND_STRENGTH = 0.6


def build_voice_profile(reference_audio_path: str) -> dict | None:
    """
    Analyse source speaker audio and return a voice profile dict with:
        mean_pitch      (Hz)
        pitch_std       (Hz)
        formant_shift   (semitones — how far to shift MMS pitch toward speaker)
        speaking_rate   (syllables/sec, approx)
        pause_ratio     (0–1)
    Returns None on failure.
    """
    try:
        import parselmouth
        import librosa

        snd      = parselmouth.Sound(reference_audio_path)
        duration = snd.get_total_duration()

        # ── Pitch ──────────────────────────────────────────────────────
        pitch_vals = snd.to_pitch().selected_array["frequency"]
        voiced     = pitch_vals[pitch_vals > 0]

        if len(voiced) < 5:
            log.warning("Too few voiced frames — using neutral defaults.")
            mean_pitch, pitch_std = 150.0, 20.0
        else:
            mean_pitch = float(np.mean(voiced))
            pitch_std  = float(np.std(voiced))

        formant_shift = float(np.clip(
            12.0 * np.log2(mean_pitch / MMS_BASE_PITCH) if mean_pitch > 0 else 0.0,
            -8.0, 8.0
        ))

        # ── Speaking rate ─────────────────────────────────────────────
        y, sr      = librosa.load(reference_audio_path, sr=None)
        intervals  = librosa.effects.split(y, top_db=30)
        speech_dur = sum((e - s) for s, e in intervals) / sr
        pause_ratio   = max(0.0, 1.0 - speech_dur / max(duration, 0.001))
        speaking_rate = round((1.0 - pause_ratio) * 4.5, 2)

        profile = {
            "mean_pitch":    round(mean_pitch, 2),
            "pitch_std":     round(pitch_std, 2),
            "formant_shift": round(formant_shift, 3),
            "speaking_rate": speaking_rate,
            "pause_ratio":   round(pause_ratio, 3),
        }
        log.info("Voice profile: %s", profile)
        return profile

    except ImportError as e:
        log.error("Missing dep: %s  →  pip install praat-parselmouth librosa", e)
        return None
    except Exception as e:
        log.error("Voice profile failed: %s", e)
        return None


def apply_voice_conversion(
    audio: np.ndarray,
    sample_rate: int,
    profile: dict,
    strength: float = BLEND_STRENGTH,
) -> np.ndarray:
    """
    Apply voice conversion to MMS-TTS waveform.
    Steps: pitch shift → time stretch → spectral EQ → normalise.
    Returns float32 mono waveform at same sample_rate.
    """
    if profile is None or strength <= 0:
        return audio

    try:
        import librosa

        converted = audio.copy()

        # ── 1. Pitch shift ────────────────────────────────────────────
        semitones = profile.get("formant_shift", 0.0) * strength
        if abs(semitones) > 0.2:
            converted = librosa.effects.pitch_shift(
                y=converted, sr=sample_rate, n_steps=semitones
            )

        # ── 2. Time stretch ───────────────────────────────────────────
        target_rate = profile.get("speaking_rate", 4.5)
        rate_ratio  = 1.0 + (float(np.clip(target_rate / 4.5, 0.78, 1.35)) - 1.0) * strength

        if abs(rate_ratio - 1.0) > 0.03:
            try:
                import pyrubberband as pyrb
                converted = pyrb.time_stretch(converted, sample_rate, rate_ratio)
            except ImportError:
                converted = librosa.effects.time_stretch(y=converted, rate=rate_ratio)

        # ── 3. Spectral EQ (timbral colour) ──────────────────────────
        pitch_std = profile.get("pitch_std", 20.0)
        gain_db = (1.5 if pitch_std > 40 else -1.0 if pitch_std < 15 else 0.0) * strength
        if abs(gain_db) > 0.1:
            converted = _boost_treble(converted, sample_rate, gain_db)

        # ── 4. Normalise ──────────────────────────────────────────────
        peak = np.max(np.abs(converted))
        if peak > 0:
            converted = converted / peak * 0.92

        return converted.astype(np.float32)

    except Exception as e:
        log.error("Voice conversion failed (returning original): %s", e)
        return audio


def _boost_treble(audio: np.ndarray, sr: int, gain_db: float) -> np.ndarray:
    """Simple high-shelf EQ via FFT — no extra dependencies."""
    spec  = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    gain  = 10 ** (gain_db / 20.0)
    mask  = np.where(freqs > 3000.0, gain, 1.0)
    return np.fft.irfft(spec * mask, n=len(audio)).astype(np.float32)
