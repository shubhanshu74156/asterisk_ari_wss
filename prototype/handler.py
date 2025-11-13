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

from agent import Agent
from adaptive_VAD import AdaptiveVAD
from utilities import Utils




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


class AdvancedHandler:
    def __init__(self, conn, addr, custom_function_prompt=""):
        self.conn = conn
        self.addr = addr
        self.utils = Utils()
        self.agent = Agent(custom_function_prompt)
        self.vad = AdaptiveVAD()  # FIXED: Use AdaptiveVAD
        self.conversation_history = []
        self.is_processing = False
        self.call_uuid = None
        self.processing_cancelled = False
        self.audio_sending_cancelled = False
        self.is_sending_audio = False
        self.audio_chunk_count = 0

        # FIXED: Rate limiting
        self.interruption_count = 0
        self.last_interruption_time = 0
        self.max_interruptions_per_second = 5

        # Create event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def cancel_audio_sending(self):
        """Cancel any ongoing audio transmission with rate limiting"""
        current_time = time.time()

        # FIXED: Rate limit interruption processing
        if current_time - self.last_interruption_time < 1.0:
            self.interruption_count += 1
            if self.interruption_count > self.max_interruptions_per_second:
                return
        else:
            self.interruption_count = 0

        self.last_interruption_time = current_time

        if self.is_sending_audio:
            print("🚫 [AUDIO] Cancelling transmission")
            self.audio_sending_cancelled = True

    def process_transcription(self, audio_data):
        """Process the audio data and generate AI response"""
        try:
            if self.processing_cancelled:
                print("🚫 [PIPELINE] Cancelled")
                return

            print(f"🔄 [PIPELINE] Processing {len(audio_data)} bytes")

            transcription = self.loop.run_until_complete(
                self.agent.transcribe_audio(audio_data)
            )

            if self.processing_cancelled:
                print("🚫 [PIPELINE] Cancelled during transcription")
                return

            if not transcription or len(transcription.strip()) == 0:
                print("⚠️ [PIPELINE] Empty transcription")
                return

            print(f"👤 [USER] '{transcription}'")

            goodbye_phrases = ['bye', 'goodbye', 'bye-bye', 'see you', 'farewell']
            if any(phrase in transcription.lower() for phrase in goodbye_phrases):
                print("👋 [GOODBYE] Ending conversation")
                farewell = "Goodbye! Have a great day!"
                farewell_audio = self.loop.run_until_complete(self.agent.text_to_speech(farewell))
                if farewell_audio:
                    self.utils.send_audio_chunks(self.conn, farewell_audio, self)
                self.conn.close()
                return

            if self.processing_cancelled:
                return

            print("🧠 [PIPELINE] Getting AI response")
            ai_response, function_results, end_call_requested = self.loop.run_until_complete(
                self.agent.get_ai_response_with_functions(transcription, self.conversation_history)
            )

            if self.processing_cancelled:
                return

            print(f"🤖 [AI] '{ai_response}'")
            if function_results:
                for result in function_results:
                    print(f"   🔧 {result.get('action', 'unknown')}: {result.get('message', 'N/A')}")

            if end_call_requested:
                print("📞 [AI-ENDING] AI ending call")
                farewell_audio = self.loop.run_until_complete(self.agent.text_to_speech(ai_response))
                if farewell_audio:
                    self.utils.send_audio_chunks(self.conn, farewell_audio, self)
                self.conn.close()
                return

            print("🔊 [PIPELINE] Converting to speech")
            response_audio = self.loop.run_until_complete(self.agent.text_to_speech(ai_response))

            if not self.processing_cancelled and response_audio:
                print("📤 [PIPELINE] Sending audio")

                self.audio_sending_cancelled = False
                self.is_sending_audio = True

                audio_sent = self.utils.send_audio_chunks(self.conn, response_audio, self)

                self.is_sending_audio = False

                if audio_sent:
                    print("✅ [PIPELINE] Response sent")
                else:
                    print("🚫 [PIPELINE] Transmission interrupted")

            elif self.processing_cancelled:
                print("🚫 [PIPELINE] Not sending - cancelled")
            else:
                print("❌ [PIPELINE] Failed to generate audio")

        except Exception as e:
            print(f"❌ [PIPELINE] Error: {e}")

    def handle_client(self):
        """Handle AudioSocket client connection"""
        print(f"🔗 [CONNECTION] {self.addr}")

        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        try:
            msg_type, payload = self.utils.read_message(self.conn)
            if msg_type == MSG_UUID and len(payload) == 16:
                self.call_uuid = payload.hex()
                print(f"✅ [UUID] {self.call_uuid[:8]}...")
            else:
                print(f"❌ [SETUP] Expected UUID, got {msg_type}")
                return

            greeting = "Hello! I'm your AI assistant with adaptive noise detection. What can I do for you?"
            print(f"👋 [GREETING] Sending greeting")

            greeting_audio = self.loop.run_until_complete(self.agent.text_to_speech(greeting))
            if greeting_audio:
                self.is_sending_audio = True
                audio_sent = self.utils.send_audio_chunks(self.conn, greeting_audio, self)
                self.is_sending_audio = False

                if audio_sent:
                    print("✅ [GREETING] Sent")
                else:
                    print("🚫 [GREETING] Interrupted")
            else:
                print("❌ [GREETING] Failed to generate")
                return

            self.conversation_history.append({"role": "assistant", "content": greeting})
            print("🎧 [READY] Listening...")

            self.audio_chunk_count = 0

            while True:
                try:
                    msg_type, payload = self.utils.read_message(self.conn)
                    if msg_type is None:
                        print("🔌 [CONNECTION] Closed")
                        break

                    if msg_type == MSG_HANGUP:
                        print("📞 [HANGUP] Received")
                        break

                    elif msg_type == MSG_AUDIO:
                        self.audio_chunk_count += 1

                        # Handle speech interruption
                        if (self.is_processing or self.is_sending_audio) and self.vad.is_speaking:
                            if self.is_processing:
                                self.processing_cancelled = True
                            if self.is_sending_audio:
                                self.cancel_audio_sending()
                            continue

                        should_process = self.vad.add_audio(payload, is_processing=self.is_processing)

                        if should_process:
                            if self.is_processing:
                                self.processing_cancelled = True
                                time.sleep(0.05)

                            if self.is_sending_audio:
                                self.cancel_audio_sending()
                                time.sleep(0.05)

                            self.is_processing = True
                            self.processing_cancelled = False
                            self.audio_sending_cancelled = False

                            audio_data = self.vad.get_audio_buffer()

                            processing_thread = threading.Thread(
                                target=self._process_audio_thread,
                                args=(audio_data,),
                                daemon=True
                            )
                            processing_thread.start()

                    elif msg_type == MSG_DTMF:
                        dtmf_digit = chr(payload[0]) if payload else ""
                        print(f"📞 [DTMF] {dtmf_digit}")
                        if dtmf_digit == '*':
                            print("⭐ [DTMF] Ending call")
                            if self.is_sending_audio:
                                self.cancel_audio_sending()
                            farewell = "Goodbye! Have a great day!"
                            farewell_audio = self.loop.run_until_complete(self.agent.text_to_speech(farewell))
                            if farewell_audio:
                                self.is_sending_audio = True
                                self.utils.send_audio_chunks(self.conn, farewell_audio, self)
                                self.is_sending_audio = False
                            break

                    elif msg_type == MSG_ERROR:
                        error_code = payload[0] if payload else 0
                        print(f"🚨 [ERROR] Code: {error_code}")
                        break

                except Exception as e:
                    print(f"❌ [LOOP] Error: {e}")
                    continue

        except Exception as e:
            print(f"💥 [HANDLER] Error: {e}")

        finally:
            print(f"🔌 [DISCONNECT] {self.addr} ({self.audio_chunk_count} chunks)")
            self.loop.close()
            self.conn.close()

    def _process_audio_thread(self, audio_data):
        """Process audio in separate thread"""
        try:
            self.process_transcription(audio_data)
        finally:
            self.is_processing = False
            self.processing_cancelled = False
            self.vad.reset_state()  # FIXED: Reset VAD state after processing