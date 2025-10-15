import wave
import numpy as np
import asyncio
import os
import tempfile
from openai import OpenAI
from dotenv import load_dotenv

# Load environment
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Agent:
    def __init__(self):
        print("🤖 Agent Class Initialized")

    async def transcribe_audio(self, audio_bytes, sample_rate=8000):
        """Async transcription using OpenAI Whisper"""
        temp_path = None
        try:
            print(f"🎙️ [TRANSCRIBE] Starting transcription of {len(audio_bytes)} bytes...")

            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_path = temp_audio.name

            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)

            print(f"💾 [TRANSCRIBE] WAV file created: {os.path.basename(temp_path)}")

            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            with open(temp_path, 'rb') as audio_file:
                print("📡 [TRANSCRIBE] Sending to OpenAI Whisper API...")
                transcript = await loop.run_in_executor(
                    None, 
                    lambda: client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en"
                    )
                )

            result = transcript.text.strip()
            print(f"✅ [TRANSCRIBE] Success: '{result}'")
            return result

        except Exception as e:
            print(f"❌ [TRANSCRIBE] Error: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    print(f"🗑️ [TRANSCRIBE] Cleaned up temp file")
                except:
                    pass

    async def get_ai_response(self, user_text, conversation_history):
        """Get AI response using OpenAI Chat API (async)"""
        try:
            print(f"🧠 [CHAT] Processing user input: '{user_text}'")

            conversation_history.append({
                "role": "user",
                "content": user_text
            })

            print("📡 [CHAT] Sending to OpenAI Chat API...")

            # Run chat completion in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=conversation_history,
                    temperature=0.7,
                    max_tokens=150
                )
            )

            ai_text = response.choices[0].message.content
            print(f"✅ [CHAT] AI Response: '{ai_text}'")

            conversation_history.append({
                "role": "assistant",
                "content": ai_text
            })

            return ai_text
        except Exception as e:
            print(f"❌ [CHAT] Error: {e}")
            return "I'm sorry, I didn't catch that. Could you please repeat?"

    async def text_to_speech(self, text):
        """Convert text to speech using OpenAI TTS (async)"""
        try:
            print(f"🔊 [TTS] Converting to speech: '{text}'")
            print("📡 [TTS] Sending to OpenAI TTS API...")

            # Run TTS in thread pool
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

            print("✅ [TTS] Received audio response from OpenAI")

            # OpenAI TTS outputs 24kHz PCM, downsample to 8kHz
            audio_data = response.content
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Downsample from 24kHz to 8kHz
            downsample_factor = 3
            downsampled = audio_array[::downsample_factor]

            result = downsampled.tobytes()
            print(f"🔄 [TTS] Downsampled audio: {len(downsampled)} samples at 8kHz ({len(result)} bytes)")

            return result
        except Exception as e:
            print(f"❌ [TTS] Error: {e}")
            return None