"""
models/model_manager.py
"""

import os
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tts.mms_tts import load_model


class ModelManager:

    def __init__(self):
        self.whisper_model    = None
        self.translator_model = None
        self.tokenizer        = None
        # Default to 'medium', but allow 'base' or 'tiny' for free tiers
        self.whisper_size = os.getenv("WHISPER_MODEL", "medium")

    def load_models(self) -> None:
        print(f"\nLoading Whisper Model ({self.whisper_size})...")
        self.whisper_model = whisper.load_model(self.whisper_size)
        print("Whisper Loaded!")

        model_name = "facebook/nllb-200-distilled-600M"
        print("\nLoading Translator...")
        self.tokenizer        = AutoTokenizer.from_pretrained(model_name)
        self.translator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Translator Loaded!")

        print("\nPreloading MMS TTS Models...")
        for lang in ["eng_Latn", "hin_Deva"]:
            try:
                load_model(lang)
                print(f"  ✓ MMS loaded: {lang}")
            except Exception as e:
                print(f"  ✗ MMS failed: {lang} — {e}")

        print("MMS Preloading Complete!")
