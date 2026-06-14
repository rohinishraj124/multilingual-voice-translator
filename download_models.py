import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, VitsModel
import torch
import os

def download():
    # 1. Whisper Medium
    print("--- Downloading Whisper medium ---")
    whisper.load_model("medium")

    # 2. NLLB-200
    nllb_model = "facebook/nllb-200-distilled-600M"
    print(f"--- Downloading {nllb_model} ---")
    AutoTokenizer.from_pretrained(nllb_model)
    AutoModelForSeq2SeqLM.from_pretrained(nllb_model)

    # 3. MMS-TTS (Top 5 common languages)
    mms_langs = ["eng", "hin", "spa", "fra", "deu"]
    for lang in mms_langs:
        mms_model = f"facebook/mms-tts-{lang}"
        print(f"--- Downloading {mms_model} ---")
        try:
            AutoTokenizer.from_pretrained(mms_model)
            VitsModel.from_pretrained(mms_model)
        except Exception as e:
            print(f"Failed to download {mms_model}: {e}")

    print("\n--- All models downloaded successfully ---")

if __name__ == "__main__":
    download()
