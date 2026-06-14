"""
stt/whisper_engine.py

FIX 3: Removed hardcoded language="hi" (Hindi) in whisper_model.transcribe().
        This forced Whisper to always treat audio as Hindi, completely breaking
        auto-detection and all other source languages (English, Spanish, etc.).
        Now language=None so Whisper auto-detects — which is its main feature.

FIX 4: Removed the intermediate librosa resample + soundfile write step.
        recorder.py now saves at 16 kHz directly (Whisper's native rate),
        making this extra I/O and dependency unnecessary.

IMPROVEMENT: Returns full result dict so callers can use segments for
             word-level timestamps if needed in the future.
"""

import whisper


def transcribe_audio(
    whisper_model: whisper.Whisper,
    audio_path: str,
    language: str | None = None   # FIX 3: None = auto-detect
) -> dict:
    """
    Transcribe audio to text using OpenAI Whisper.

    Args:
        whisper_model: Loaded Whisper model instance.
        audio_path:    Path to .wav audio file (16 kHz recommended).
        language:      BCP-47 language code to force (e.g. "en"), or
                       None to let Whisper auto-detect.

    Returns:
        {"text": str, "language": str, "segments": list}
    """
    result = whisper_model.transcribe(
        audio_path,
        language=language,      # FIX 3
        fp16=False,
        beam_size=5,
        temperature=0,
        no_speech_threshold=0.6,
        condition_on_previous_text=False
    )

    return {
        "text": result["text"].strip(),
        "language": result.get("language", "en"),
        "segments": result.get("segments", [])
    }
