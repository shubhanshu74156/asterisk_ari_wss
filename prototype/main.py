
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

from handler import AdvancedHandler


# Load environment
load_dotenv()
print("🌍 Environment Variables Loaded")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
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




def main():
    print(f"🚀 Adaptive Noise Detection AI AudioSocket Server")
    print(f"🌐 {HOST}:{PORT}")
    print(f"🎯 Features: Adaptive VAD, Smart Interruption, Function Calling")

    custom_prompt = """
    Be proactive with functions when helpful:
    - Use dice/coins for games and decisions  
    - Suggest animals for education and fun
    - Generate numbers for games
    - Keep responses concise for interruption-friendly conversation
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"✅ Server ready!")

        connection_count = 0

        while True:
            try:
                conn, addr = s.accept()
                connection_count += 1
                print(f"🎉 Connection #{connection_count} from {addr}")

                handler = AdvancedHandler(conn, addr, custom_prompt)

                handler_thread = threading.Thread(
                    target=handler.handle_client, 
                    daemon=True,
                    name=f"Handler-{connection_count}"
                )
                handler_thread.start()

            except KeyboardInterrupt:
                print("\n⛔ Shutting down...")
                break
            except Exception as e:
                print(f"💥 Server error: {e}")


if __name__ == "__main__":
    main()
