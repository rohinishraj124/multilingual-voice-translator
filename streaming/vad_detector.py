import queue
import numpy as np


class VADDetector:

    def __init__(
        self,
        sample_rate=16000,
        silence_duration=1.0,
        speech_threshold=0.01,
        min_speech_duration=0.8
    ):

        self.sample_rate = sample_rate

        self.silence_duration = silence_duration

        self.speech_threshold = speech_threshold

        self.min_speech_duration = min_speech_duration

        self.speech_queue = queue.Queue()

        self.current_segment = []

        self.silence_chunks = 0

    def process_chunk(self, chunk):

        # Calculate RMS volume
        rms = np.sqrt(
            np.mean(chunk ** 2)
        )

        # Speech detected
        if rms > self.speech_threshold:

            self.current_segment.append(
                chunk.flatten()
            )

            self.silence_chunks = 0

            return

        # No active speech segment
        if len(self.current_segment) == 0:
            return

        # Silence detected after speech
        self.silence_chunks += 1

        silence_time = (
            self.silence_chunks *
            len(chunk)
        ) / self.sample_rate

        # End segment after enough silence
        if silence_time >= self.silence_duration:

            segment = np.concatenate(
                self.current_segment
            )

            duration = (
                len(segment) /
                self.sample_rate
            )

            # Ignore tiny segments
            if duration >= self.min_speech_duration:

                self.speech_queue.put(
                    segment
                )

                print(
                    f"Speech segment emitted "
                    f"({duration:.2f}s)"
                )

            else:

                print(
                    f"Ignored short segment "
                    f"({duration:.2f}s)"
                )

            # Reset for next segment
            self.current_segment = []

            self.silence_chunks = 0

    def get_segment(self):

        return self.speech_queue.get()