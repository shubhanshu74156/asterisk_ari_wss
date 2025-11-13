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


class Utils:
    def __init__(self):
        print("🔧 Utils Class Initialized")

    def send_audio_chunks(self, conn, audio_data, handler_ref=None):
        """Send audio data in 320-byte chunks via AudioSocket protocol"""
        chunk_size = 320
        total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size

        print(f"📤 [SEND] Transmitting {len(audio_data)} bytes ({total_chunks} chunks)")

        chunks_sent = 0
        batch_size = 10  # Check interruption every 10 chunks

        for batch_start in range(0, len(audio_data), chunk_size * batch_size):
            # Check for interruption at batch level
            if handler_ref and handler_ref.audio_sending_cancelled:
                remaining = total_chunks - chunks_sent
                print(f"🚫 [SEND] Cancelled at chunk {chunks_sent}/{total_chunks} ({remaining} skipped)")
                return False

            batch_end = min(batch_start + chunk_size * batch_size, len(audio_data))
            for i in range(batch_start, batch_end, chunk_size):
                chunk = audio_data[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk += b'\x00' * (chunk_size - len(chunk))
                header = struct.pack('B', MSG_AUDIO) + struct.pack('>H', len(chunk))

                try:
                    conn.sendall(header + chunk)
                    chunks_sent += 1
                except Exception as e:
                    print(f"❌ Error at chunk {chunks_sent}: {e}")
                    return False

            time.sleep(0.01)  # Reduced sleep

            if DEBUG and chunks_sent % 50 == 0:
                print(f"📡 Progress: {chunks_sent}/{total_chunks}")

        print(f"✅ [SEND] Complete ({total_chunks} chunks)")
        return True

    def read_message(self, conn):
        """Read one full AudioSocket message"""
        try:
            header = conn.recv(3)
            if len(header) < 3:
                return None, None

            msg_type = header[0]
            payload_length = struct.unpack('>H', header[1:3])[0]

            if DEBUG and msg_type != MSG_AUDIO:
                msg_types = {MSG_HANGUP: "HANGUP", MSG_UUID: "UUID", MSG_DTMF: "DTMF", MSG_ERROR: "ERROR"}
                msg_name = msg_types.get(msg_type, f'UNKNOWN({msg_type})')
                print(f"📥 [READ] {msg_name}: {payload_length} bytes")

            payload = b''
            while len(payload) < payload_length:
                chunk = conn.recv(payload_length - len(payload))
                if not chunk:
                    break
                payload += chunk

            return msg_type, payload
        except Exception as e:
            print(f"❌ Error reading message: {e}")
            return None, None


