import numpy as np
import time

DEBUG = True

class SimpleVAD:
    """Simple Voice Activity Detection using RMS and silence detection"""

    def __init__(self):
        self.audio_buffer = bytearray()
        self.silence_threshold = 200  # RMS threshold for silence
        self.speech_threshold = 500   # RMS threshold for speech
        self.min_speech_duration = 1.0  # seconds
        self.silence_duration = 1.0     # seconds of silence to trigger processing
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        print("🎤 Simple VAD initialized")

    def calculate_rms(self, audio_data):
        """Calculate RMS of audio data"""
        if len(audio_data) == 0:
            return 0.0
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) == 0:
                return 0.0
            rms = float(np.sqrt(np.mean(audio_array.astype(np.float64)**2)))
            return rms if not np.isnan(rms) else 0.0
        except:
            return 0.0

    def add_audio(self, audio_chunk):
        """Add audio chunk and return speech status"""
        self.audio_buffer.extend(audio_chunk)

        # Calculate RMS of the chunk
        rms = self.calculate_rms(audio_chunk)
        current_time = time.time()

        # Determine if currently speaking
        if rms > self.speech_threshold:
            if not self.is_speaking:
                print("🎙️ [VAD] 🔴 Speech detected - Recording started")
                self.is_speaking = True
                self.speech_start_time = current_time
            self.last_speech_time = current_time

        elif self.is_speaking and rms < self.silence_threshold:
            # Check if we've been silent long enough
            silence_duration = current_time - self.last_speech_time
            if silence_duration >= self.silence_duration:
                speech_duration = self.last_speech_time - self.speech_start_time

                if speech_duration >= self.min_speech_duration:
                    print(f"🎙️ [VAD] ⏹️ Speech ended after {speech_duration:.1f}s - Processing...")
                    self.is_speaking = False
                    return True  # Signal to process speech
                else:
                    print(f"🎙️ [VAD] ⚠️ Speech too short ({speech_duration:.1f}s) - Ignoring")
                    self.is_speaking = False
                    self.audio_buffer.clear()

        # Log RMS periodically
        if DEBUG and len(self.audio_buffer) % 16000 == 0:  # Every ~1 second of audio
            duration = len(self.audio_buffer) / (8000 * 2)
            print(f"📊 [VAD] Buffer: {duration:.1f}s, RMS: {rms:.0f}, Speaking: {self.is_speaking}")

        return False

    def get_audio_buffer(self):
        """Get and clear the audio buffer"""
        audio_data = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        return audio_data
