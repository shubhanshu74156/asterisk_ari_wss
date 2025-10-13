import os
import socket
import sounddevice as sd
import numpy as np
import threading
import wave
import io
import struct
import asyncio
import tempfile
import time
from collections import deque
from openai import OpenAI
from dotenv import load_dotenv

# Load environment
load_dotenv()

load_dotenv()

HOST = "0.0.0.0"
PORT = 9092

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Helper:
    
    def __init__(self):
        pass

    def normalize_audio(self, audio_data):
        """Normalize audio to increase volume"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            if len(audio_array) == 0:
                return audio_data

            # Calculate current peak
            peak = np.max(np.abs(audio_array))
            if peak > 0:
                # Normalize to 80% of max to avoid clipping
                target_peak = 32768 * 0.8
                gain = target_peak / peak
                # Limit gain to avoid amplifying noise too much
                gain = min(gain, 10.0)
                audio_array = audio_array * gain
                audio_array = np.clip(audio_array, -32768, 32767)

            return audio_array.astype(np.int16).tobytes()
        except Exception as e:
            print(f"[!] Normalization error: {e}")
            return audio_data

class Agent:

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.chunk = 1024  # Number of samples per packet
        self.channels = 1
        self.samplerate = 44100
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server.bind((self.host, self.port))
        self.helper = Helper()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print(f"UDP Audio Server listening on {self.host}:{self.port}")

    def receive(self):
        with sd.OutputStream(samplerate=self.samplerate, channels=self.channels, dtype='int16') as stream:
            while True:
                data, addr = self.server.recvfrom(self.chunk * 2)  # 2 bytes per sample for int16
                audio_data = np.frombuffer(data, dtype='int16')
                stream.write(audio_data)

        self.transcribe(audio_data=audio_data)

    def transcribe(self, audio_data, sample_rate=8000):
        temp_path = None
        try:
            normalized_audio = self.helper.normalize_audio(audio_data)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_path = temp_audio.name
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(normalized_audio)
            with open(temp_path, 'rb') as audio_file:
                transcript = asyncio.to_thread(
                    self.client.audio.transcriptions.create,
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            text = transcript.text.strip()
            return self.chat(text)
        except Exception as e:
            print(f"[!] Transcription error: {e}")
            import traceback; traceback.print_exc()
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    def chat(self, text):
        pass

    def speak(self, text):
        pass

if __name__ == "__main__":
    agent = Agent(HOST, PORT)
    agent.receive()
