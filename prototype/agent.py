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

from fucntion_library import FunctionLibrary


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


class Agent:
    def __init__(self, custom_function_prompt=""):
        self.function_library = FunctionLibrary()
        self.custom_function_prompt = custom_function_prompt
        self.system_message_cache = None
        print("🤖 Agent Class Initialized with AI-driven function calling")

    async def transcribe_audio(self, audio_bytes, sample_rate=8000):
        """Async transcription using OpenAI Whisper"""
        temp_path = None
        try:
            print(f"🎙️ [TRANSCRIBE] Processing {len(audio_bytes)} bytes")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_path = temp_audio.name

            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)

            loop = asyncio.get_event_loop()
            with open(temp_path, 'rb') as audio_file:
                print("📡 [TRANSCRIBE] -> Whisper API")
                transcript = await loop.run_in_executor(
                    None, 
                    lambda: client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en"
                    )
                )

            result = transcript.text.strip()
            print(f"✅ [TRANSCRIBE] '{result}'")
            return result

        except Exception as e:
            print(f"❌ [TRANSCRIBE] Error: {e}")
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

    async def get_ai_response_with_functions(self, user_text, conversation_history):
        """Get AI response with intelligent function calling"""
        try:
            print(f"🧠 [CHAT] Processing: '{user_text}'")

            if not self.system_message_cache:
                self.system_message_cache = {
                    "role": "system",
                    "content": f"""You are a helpful AI phone assistant. Keep responses brief and conversational (under 3 sentences).

Available functions:
- Roll dice for games and random choices
- Pick random animals for education and games  
- Generate random numbers for games
- Flip coins for decisions
- End calls when appropriate

{self.custom_function_prompt}

Respond naturally and incorporate function results."""
                }

            if not conversation_history or conversation_history[0]["role"] != "system":
                conversation_history.insert(0, self.system_message_cache)
            else:
                conversation_history[0] = self.system_message_cache

            conversation_history.append({"role": "user", "content": user_text})

            print("📡 [CHAT] -> OpenAI API")

            available_functions = self.function_library.get_available_functions()

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=conversation_history,
                    temperature=0.7,
                    max_tokens=150,
                    tools=available_functions,
                    tool_choice="auto"
                )
            )

            message = response.choices[0].message

            if message.tool_calls:
                print(f"🔧 [FUNCTIONS] Calling {len(message.tool_calls)} function(s)")

                conversation_history.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [tc.dict() for tc in message.tool_calls]
                })

                function_results = []
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"🎯 [FUNCTION] {function_name}({function_args})")

                    result = self.function_library.execute_function(function_name, function_args)
                    function_results.append(result)

                    conversation_history.append({
                        "role": "tool",
                        "content": json.dumps(result),
                        "tool_call_id": tool_call.id
                    })

                print("🔄 [CHAT] -> Final response")
                final_response = await loop.run_in_executor(
                    None,
                    lambda: client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=conversation_history,
                        temperature=0.7,
                        max_tokens=150
                    )
                )

                ai_text = final_response.choices[0].message.content
                print(f"✅ [CHAT] '{ai_text}'")

                end_call_requested = any(result.get("action") == "end_call" for result in function_results)

                return ai_text, function_results, end_call_requested

            else:
                ai_text = message.content
                print(f"✅ [CHAT] '{ai_text}'")

                conversation_history.append({"role": "assistant", "content": ai_text})
                return ai_text, [], False

        except Exception as e:
            print(f"❌ [CHAT] Error: {e}")
            return "Sorry, could you repeat that?", [], False

    async def text_to_speech(self, text):
        """Convert text to speech using OpenAI TTS"""
        try:
            print(f"🔊 [TTS] '{text}'")
            print("📡 [TTS] -> OpenAI TTS")

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=text,
                    response_format="pcm",
                    speed=1.0
                )
            )

            print("✅ [TTS] Audio received")

            audio_data = response.content
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Downsample from 24kHz to 8kHz
            downsampled = audio_array[::3]
            result = downsampled.tobytes()

            print(f"🔄 [TTS] {len(downsampled)} samples ({len(result)} bytes)")
            return result

        except Exception as e:
            print(f"❌ [TTS] Error: {e}")
            return None
