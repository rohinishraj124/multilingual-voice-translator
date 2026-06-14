import queue
import time
import threading

from streaming.audio_stream import AudioStream
from streaming.vad_detector import VADDetector

from streaming.worker import (
    STTWorker,
    TranslationWorker,
    TTSWorker
)


class StreamManager:

    def __init__(
        self,
        model_manager,
        source_language, # ADDED THIS
        target_language
    ):

        self.model_manager = model_manager
        self.source_language = source_language # ADDED THIS
        self.target_language = target_language
        
        # ... (rest of your variables stay the same)
        self.stream = None
        self.vad = None
        self.stt_worker = None
        self.translation_worker = None
        self.tts_worker = None
        self.running = False
        self.transcript_queue = queue.Queue()
        self.translation_display_queue = queue.Queue()
        self.process_thread = None

    def start_streaming(self):

        if self.running:
            return

        self.stream = AudioStream()

        self.vad = VADDetector(
            silence_duration=1.5,
            speech_threshold=0.01,
            min_speech_duration=0.8
        )

        self.stt_worker = STTWorker(
                self.model_manager.whisper_model,
                self.vad.speech_queue,
                self.source_language
            )

        self.translation_worker = (
            TranslationWorker(
                self.model_manager.translator_model,
                self.model_manager.tokenizer,
                self.stt_worker.text_queue,
                self.target_language
            )
        )

        self.tts_worker = TTSWorker(
            self.translation_worker.translation_queue,
            self.target_language
        )

        self.transcript_queue = (
            self.stt_worker.display_queue
        )

        self.translation_display_queue = (
            self.translation_worker.display_queue
        )

        self.stt_worker.start()

        self.translation_worker.start()

        self.tts_worker.start()

        self.stream.start()

        self.running = True
        
        # ADDED: Start the background processing thread
        self.process_thread = threading.Thread(
            target=self._process_loop, 
            daemon=True
        )
        self.process_thread.start()

    # ADDED: Background loop to continuously fetch audio chunks
    def _process_loop(self):
        while self.running:
            self.process()

    def process(self):

        if not self.running:
            return

        # UPDATED: We will pass a timeout to prevent permanent blocking
        chunk = self.stream.get_chunk(timeout=0.5)
        
        # ADDED: Handle the case where no audio chunk was received within the timeout
        if chunk is None:
            return

        if (
            not self.tts_worker.is_speaking
            and
            time.time()
            - self.tts_worker.last_tts_time
            > 1
        ):

            self.vad.process_chunk(
                chunk
            )

    def stop_streaming(self):

        if not self.running:
            return
            
        # UPDATED: Set running to False before stopping workers so loops break
        self.running = False

        self.stt_worker.stop()

        self.translation_worker.stop()

        self.tts_worker.stop()

        self.stream.stop()
        
        # ADDED: Wait for the process loop to close cleanly
        if self.process_thread:
            self.process_thread.join(timeout=2.0)