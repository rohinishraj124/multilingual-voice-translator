"""
translation/translator.py

FIX 5: max_length=50 was too short for anything more than ~10 words,
        silently truncating translated output. Increased to 512 (NLLB's
        recommended max) with num_beams for quality.

IMPROVEMENT: Added num_beams and early_stopping for better translation
             quality with minimal speed cost on CPU.
"""


def translate_text(
    text: str,
    tokenizer,
    translator_model,
    source_lang: str,
    target_lang: str
) -> str:
    """
    Translate text from source_lang to target_lang using NLLB-200.

    Args:
        text:             Source text to translate.
        tokenizer:        NLLB AutoTokenizer instance.
        translator_model: NLLB AutoModelForSeq2SeqLM instance.
        source_lang:      NLLB language code e.g. "eng_Latn".
        target_lang:      NLLB language code e.g. "hin_Deva".

    Returns:
        Translated string, or "" if input is empty.
    """
    if not text.strip():
        return ""

    tokenizer.src_lang = source_lang

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

    translated_tokens = translator_model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=512,     # FIX 5: was 50, silently cut long sentences
        num_beams=4,        # beam search for better quality
        early_stopping=True
    )

    translated_text = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )

    return translated_text[0]
