import queue
import threading
import numpy as np
import time
from tts.mms_tts import speak


class STTWorker(threading.Thread):

    def __init__(
        self,
        whisper_model,
        speech_queue,
        source_language=None # ADDED
    ):
        super().__init__(daemon=True)
        self.whisper_model = whisper_model
        self.speech_queue = speech_queue
        self.source_language = source_language # e.g. "hin_Deva"
        self.text_queue = queue.Queue()
        self.display_queue = queue.Queue()
        self.running = True
        
        # Mapping NLLB codes to Whisper 2-letter codes
        self.nllb_to_whisper = {
            "eng_Latn": "en", "hin_Deva": "hi", "urd_Arab": "ur", 
            "spa_Latn": "es", "fra_Latn": "fr"
        }
        self.whisper_to_nllb = {v: k for k, v in self.nllb_to_whisper.items()}

    def run(self):
        print("STT Worker Started")
        while self.running:
            try:
                segment = self.speech_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                audio = segment.astype(np.float32)

                # FIX: Build arguments for Whisper. If we know the language, force it!
                kwargs = {"fp16": False}
                if self.source_language:
                    whisper_lang = self.nllb_to_whisper.get(self.source_language)
                    if whisper_lang:
                        kwargs["language"] = whisper_lang

                result = self.whisper_model.transcribe(audio, **kwargs)
                text = result["text"].strip()

                # Determine the source language to send to the Translator
                if self.source_language:
                    nllb_src = self.source_language
                else:
                    # If "Auto Detect" was used, grab Whisper's detected language
                    det_lang = result.get("language", "en")
                    nllb_src = self.whisper_to_nllb.get(det_lang, "eng_Latn")

                if text:
                    print(f"STT: {text} (Detected Lang: {nllb_src})")
                    
                    # FIX: Put BOTH the text AND the language in the queue for the translator
                    self.text_queue.put((text, nllb_src))
                    self.display_queue.put(text)

            except Exception as e:
                print("STT Error:", e)

    def stop(self):
        self.running = False

class TranslationWorker(threading.Thread):

    def __init__(
        self,
        translator_model,
        tokenizer,
        text_queue,
        target_language
    ):
        super().__init__(daemon=True)
        self.translator_model = translator_model
        self.tokenizer = tokenizer
        self.text_queue = text_queue
        self.target_language = target_language
        self.translation_queue = queue.Queue()
        self.display_queue = queue.Queue()
        self.running = True

    def run(self):
        print("Translation Worker Started")
        while self.running:
            try:
                # FIX: Unpack the tuple sent by STT Worker
                text, src_lang = self.text_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                # CRITICAL FIX: Tell NLLB the language of the input text!
                self.tokenizer.src_lang = src_lang

                inputs = self.tokenizer(
                    text,
                    return_tensors="pt"
                )

                translated_tokens = (
                    self.translator_model.generate(
                        **inputs,
                        forced_bos_token_id=
                        self.tokenizer.convert_tokens_to_ids(
                            self.target_language
                        ),
                        max_length=256
                    )
                )

                translated_text = (
                    self.tokenizer.batch_decode(
                        translated_tokens,
                        skip_special_tokens=True
                    )[0]
                )

                print(f"Translation: {translated_text}")
                self.translation_queue.put(translated_text)
                self.display_queue.put(translated_text)

            except Exception as e:
                print("Translation Error:", e)

    def stop(self):
        self.running = False


class TTSWorker(threading.Thread):

    def __init__(
        self,
        translation_queue,
        target_language
    ):

        super().__init__(daemon=True)

        self.translation_queue = translation_queue

        self.target_language = target_language

        self.running = True

        self.is_speaking = False

        self.last_tts_time = 0

    def run(self):

        print(
            "TTS Worker Started"
        )

        while self.running:

            # UPDATED: Added timeout and try-except
            try:
                translated_text = (
                    self.translation_queue.get(timeout=1.0)
                )
            except queue.Empty:
                continue

            try:

                self.is_speaking = True

                print(
                    f"TTS: {translated_text}"
                )

                speak(
                    translated_text,
                    self.target_language
                )

                self.last_tts_time = (
                    time.time()
                )

                self.is_speaking = False

            except Exception as e:

                self.is_speaking = False

                print(
                    "TTS Error:",
                    e
                )

    def stop(self):

        self.running = False