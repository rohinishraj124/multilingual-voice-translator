"""
tts/language_manager.py

IMPROVEMENT: Expanded NLLB→MMS mapping to include more languages so the
             translator can be extended without touching core code.
"""

NLLB_TO_MMS: dict[str, str] = {
    "eng_Latn": "eng",
    "hin_Deva": "hin",
    "urd_Arab": "urd",
    "spa_Latn": "spa",
    "fra_Latn": "fra",
    # Extended
    "deu_Latn": "deu",   # German
    "por_Latn": "por",   # Portuguese
    "ara_Arab": "ara",   # Arabic
    "zho_Hans": "cmn",   # Mandarin Chinese (simplified)
    "jpn_Jpan": "jpn",   # Japanese
    "rus_Cyrl": "rus",   # Russian
    "ben_Beng": "ben",   # Bengali
}

# Whisper language code → NLLB language code
WHISPER_TO_NLLB: dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ur": "urd_Arab",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "pt": "por_Latn",
    "ar": "ara_Arab",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ru": "rus_Cyrl",
    "bn": "ben_Beng",
}


def get_mms_language(nllb_code: str) -> str:
    if nllb_code not in NLLB_TO_MMS:
        raise ValueError(
            f"No MMS mapping for '{nllb_code}'. "
            f"Available: {list(NLLB_TO_MMS.keys())}"
        )
    return NLLB_TO_MMS[nllb_code]
