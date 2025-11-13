import socket
import threading
import wave
import struct
import numpy as np
import asyncio
import os
import tempfile
import time
from openai import OpenAI
from dotenv import load_dotenv
import random
import json
import statistics
from collections import deque


# Load environment
load_dotenv()
print("🌍 Environment Variables Loaded")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("🤖 OpenAI Client Initialized")

DEBUG = False  # FIXED: Set to False for production

HOST = "0.0.0.0"
PORT = 9092

# AudioSocket message types
MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10
MSG_ERROR = 0xff

class AdaptiveVAD:
    """Enhanced Voice Activity Detection with adaptive noise thresholding"""

    def __init__(self):
        self.audio_buffer = bytearray()

        # FIXED: More conservative initial thresholds
        self.silence_threshold = 300   # Initial silence threshold
        self.speech_threshold = 600    # Initial speech threshold

        self.min_speech_duration = 0.4  # FIXED: Increased from 0.2s to reduce false positives
        self.silence_duration = 1.2     # FIXED: Increased from 1.5s for stability
        self.hangover_time = 0.3        # Seconds to maintain speech after drop

        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None  # FIXED: Initialize as None instead of time.time()
        self.processing_interrupted = False
        self.interrupted_buffer = bytearray()

        # FIXED: Reduced window size for faster adaptation
        self.rms_window = deque(maxlen=100)  # Store recent RMS values

        # FIXED: Add debouncing
        self.speech_samples = deque(maxlen=5)
        self.consecutive_speech = 0
        self.consecutive_silence = 0

        # FIXED: Track calibration
        self.calibration_complete = False
        self.calibration_samples = 0
        self.calibration_required = 20  # Number of samples needed for initial calibration

        print("🎤 Adaptive VAD initialized with smart noise detection")

    def calculate_rms(self, audio_data):
        """Calculate RMS of audio data efficiently"""
        if not audio_data or len(audio_data) == 0:
            return 0.0
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if audio_array.size == 0:
                return 0.0

            # FIXED: Use more efficient calculation
            rms = float(np.sqrt(np.mean(np.square(audio_array, dtype=np.float32))))
            return rms if not np.isnan(rms) and not np.isinf(rms) else 0.0
        except Exception as e:
            print(f"❌ [VAD] RMS calculation error: {e}")
            return 0.0

    def update_thresholds(self):
        """Adapt speech/silence thresholds based on background noise"""
        if len(self.rms_window) < 10:
            return  # Not enough data yet

        try:
            # FIXED: Use percentile-based approach instead of median
            rms_list = list(self.rms_window)

            # Calculate noise floor (10th percentile of recent RMS values)
            noise_floor = np.percentile(rms_list, 10)

            # Calculate speech level (90th percentile)
            speech_level = np.percentile(rms_list, 90)

            # FIXED: More conservative threshold calculation
            # Silence threshold: 150% of noise floor, with minimum
            self.silence_threshold = max(noise_floor * 1.5, 250)

            # Speech threshold: Between silence and speech level, with minimum
            self.speech_threshold = max(noise_floor * 3.0, 500)

            # FIXED: Ensure speech threshold is always higher than silence
            if self.speech_threshold <= self.silence_threshold:
                self.speech_threshold = self.silence_threshold * 1.5

            # FIXED: Cap maximum thresholds to prevent runaway adaptation
            self.silence_threshold = min(self.silence_threshold, 1000)
            self.speech_threshold = min(self.speech_threshold, 2000)

            # Mark calibration complete
            if not self.calibration_complete and len(self.rms_window) >= self.calibration_required:
                self.calibration_complete = True
                print(f"✅ [VAD] Calibration complete - Silence: {self.silence_threshold:.0f}, Speech: {self.speech_threshold:.0f}")

        except Exception as e:
            print(f"❌ [VAD] Threshold update error: {e}")

    def detect_spike(self, current_rms):
        """FIXED: Simplified spike detection"""
        if len(self.rms_window) < 5:
            return False

        try:
            recent_avg = np.mean(list(self.rms_window)[-5:])

            # Spike if current RMS is significantly higher than recent average
            if current_rms > recent_avg * 2.5 and current_rms > self.speech_threshold:
                return True
            return False
        except:
            return False

    def add_audio(self, audio_chunk, is_processing=False):
        """Add audio chunk and return speech status with adaptive detection"""
        self.audio_buffer.extend(audio_chunk)
        rms = self.calculate_rms(audio_chunk)

        # Add to RMS window for adaptive thresholding
        self.rms_window.append(rms)

        # Update adaptive thresholds
        self.update_thresholds()

        current_time = time.time()

        # FIXED: Initialize last_speech_time on first valid RMS
        if self.last_speech_time is None:
            self.last_speech_time = current_time

        # Add to debouncing window
        self.speech_samples.append(rms)
        avg_rms = np.mean(list(self.speech_samples))

        # FIXED: Debounced speech detection
        if avg_rms > self.speech_threshold or self.detect_spike(rms):
            self.consecutive_speech += 1
            self.consecutive_silence = 0

            # FIXED: Require 2 consecutive detections to confirm speech
            if self.consecutive_speech >= 2:
                if not self.is_speaking:
                    if is_processing:
                        print("🔄 [VAD] Speech during processing - interrupting")
                        self.processing_interrupted = True
                        self.interrupted_buffer.extend(self.audio_buffer)
                    else:
                        print("🎙️ [VAD] Speech started")
                    self.is_speaking = True
                    self.speech_start_time = current_time

                self.last_speech_time = current_time

        elif avg_rms < self.silence_threshold:
            self.consecutive_silence += 1
            self.consecutive_speech = 0

            # FIXED: Require 3 consecutive silence detections
            if self.consecutive_silence >= 3:
                # Apply hangover time
                if self.is_speaking and (current_time - self.last_speech_time) > self.hangover_time:
                    silence_duration = current_time - self.last_speech_time

                    if silence_duration >= self.silence_duration:
                        if self.speech_start_time:
                            speech_duration = self.last_speech_time - self.speech_start_time
                        else:
                            speech_duration = 0

                        if speech_duration >= self.min_speech_duration:
                            if self.processing_interrupted:
                                print(f"🎙️ [VAD] Combined speech ended ({speech_duration:.1f}s)")
                            else:
                                print(f"🎙️ [VAD] Speech ended ({speech_duration:.1f}s)")
                            self.is_speaking = False
                            self.consecutive_speech = 0  # Reset
                            self.consecutive_silence = 0  # Reset
                            return True  # Signal to process
                        else:
                            print(f"🎙️ [VAD] Speech too short ({speech_duration:.1f}s) - ignoring")
                            self.is_speaking = False
                            if not is_processing:
                                self.audio_buffer.clear()
                            self.consecutive_speech = 0
                            self.consecutive_silence = 0

        # FIXED: Periodic logging with adaptive thresholds
        if DEBUG and len(self.audio_buffer) % 32000 == 0:  # Every ~2 seconds
            duration = len(self.audio_buffer) / 16000
            status = "Speaking" if self.is_speaking else "Silent"
            if is_processing and self.is_speaking:
                status = "Interrupting"
            elif is_processing:
                status = "Processing"

            calib_status = "✓" if self.calibration_complete else "..."
            print(f"📊 [VAD] {duration:.1f}s, RMS:{avg_rms:.0f}, S_Thr:{self.silence_threshold:.0f}, P_Thr:{self.speech_threshold:.0f}, Cal:{calib_status}, {status}")

        return False

    def get_audio_buffer(self):
        """Get and clear the audio buffer, combining with interrupted audio if any"""
        if self.processing_interrupted and len(self.interrupted_buffer) > 0:
            combined_audio = bytes(self.interrupted_buffer) + bytes(self.audio_buffer)
            print(f"🔗 [VAD] Combined: {len(self.interrupted_buffer)} + {len(self.audio_buffer)} = {len(combined_audio)} bytes")
            self.interrupted_buffer.clear()
        else:
            combined_audio = bytes(self.audio_buffer)

        self.audio_buffer.clear()
        self.processing_interrupted = False
        return combined_audio

    def reset_state(self):
        """FIXED: Add method to reset speech detection state"""
        self.is_speaking = False
        self.consecutive_speech = 0
        self.consecutive_silence = 0