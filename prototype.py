
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


# Load environment
load_dotenv()
print("🌍 Environment Variables Loaded")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("🤖 OpenAI Client Initialized")

DEBUG = True

HOST = "0.0.0.0"
PORT = 9092

# AudioSocket message types
MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10
MSG_ERROR = 0xff

class Fucntion:

    def __init__(self):
        print("Funtion is initiated")

    def end_call(self, conn):
        print("End call function initiated")
        conn.close()
        print("End call function executed")

    def print_random_animal_name(self):
        print("Random animal name function initiated")
        animals = ["Lion", "Elephant", "Cheeta", "Dog", "Cat", "Pig", "Sparrow"]
        animal = random.choice(animals)
        print("Animal Selected: ", animal)
        print("Animal function executed")

    def roll_dice(self):
        print("Roll dice fucntion initiated")
        numbers = [1,2,3,4,5,6]
        number = random.choice(numbers)
        print("Number slected: ", number)
        print("Roll dice executed")



class SimpleVAD:
    """Simple Voice Activity Detection using RMS and silence detection"""

    def __init__(self):
        self.audio_buffer = bytearray()
        self.silence_threshold = 200  # RMS threshold for silence
        self.speech_threshold = 500   # RMS threshold for speech
        self.min_speech_duration = 1.0  # seconds
        self.silence_duration = 1.5     # seconds of silence to trigger processing
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        print("🎤 Simple VAD initialized")
        self.processing_interrupted = False

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

    def add_audio(self, audio_chunk, is_processing=False):
        """Add audio chunk and return speech status"""
        self.audio_buffer.extend(audio_chunk)

        # Calculate RMS of the chunk
        rms = self.calculate_rms(audio_chunk)
        current_time = time.time()

        # Determine if currently speaking
        if rms > self.speech_threshold:
            if is_processing:
                    print("🎙️ [VAD] 🔴 Speech detected during processing - Accumulating audio")
                    self.processing_interrupted = True
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


class Utils:
    def __init__(self):
        print("🔧 Utils Class Initialized")

    def send_audio_chunks(self, conn, audio_data):
        """Send audio data in 320-byte chunks via AudioSocket protocol"""
        chunk_size = 320
        total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size

        print(f"📤 [SEND] Sending {len(audio_data)} bytes in {total_chunks} chunks")

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk += b'\x00' * (chunk_size - len(chunk))
            header = struct.pack('B', MSG_AUDIO) + struct.pack('>H', len(chunk))
            try:
                conn.sendall(header + chunk)
                time.sleep(0.02)

                if DEBUG and i % (chunk_size * 10) == 0:  # Every 10th chunk
                    print(f"📡 Sent chunk {i//chunk_size + 1}/{total_chunks}")

            except Exception as e:
                print(f"❌ Error sending audio chunk {i//chunk_size + 1}: {e}")
                break

        print(f"✅ [SEND] Audio transmission complete ({total_chunks} chunks)")

    def read_message(self, conn):
        """Read one full AudioSocket message"""
        try:
            header = conn.recv(3)
            if len(header) < 3:
                return None, None

            msg_type = header[0]
            payload_length = struct.unpack('>H', header[1:3])[0]

            if DEBUG:
                msg_types = {
                    MSG_HANGUP: "HANGUP",
                    MSG_UUID: "UUID", 
                    MSG_DTMF: "DTMF",
                    MSG_AUDIO: "AUDIO",
                    MSG_ERROR: "ERROR"
                }
                msg_name = msg_types.get(msg_type, f'UNKNOWN({msg_type})')
                if msg_type != MSG_AUDIO:  # Only log non-audio messages
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


class SimpleHandler:
    def __init__(self, conn, addr):
        self.conn = conn
        self.addr = addr
        self.utils = Utils()
        self.agent = Agent()
        self.vad = SimpleVAD()
        self.functions = Fucntion()
        self.conversation_history = [
            {
                "role": "system",
                "content": "You are a helpful AI phone assistant. Keep your responses brief, clear, and conversational. Use simple language suitable for phone conversations. Keep responses under 3 sentences."
            }
        ]
        self.is_processing = False
        self.call_uuid = None
        self.processing_cancelled = False

        # Create event loop for async operations
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def process_transcription(self, audio_data):
        """Process the audio data and generate AI response"""
        try:
            print(f"🔄 [PIPELINE] Processing {len(audio_data)} bytes of audio...")

            # Transcribe audio
            transcription = self.loop.run_until_complete(
                self.agent.transcribe_audio(audio_data)
            )

            if not transcription or len(transcription.strip()) == 0:
                print("⚠️ [PIPELINE] Empty transcription received")
                return

            print(f"👤 [USER] Transcription: \"{transcription}\"\n")

            # Check for goodbye phrases
            goodbye_phrases = ['bye', 'goodbye', 'bye-bye', 'see you', 'farewell']
            if any(phrase in transcription.lower() for phrase in goodbye_phrases):
                print("👋 [GOODBYE] Goodbye detected, ending conversation...")
                farewell = "Goodbye! Have a great day!"
                print(f"🤖 [AI] Final response: '{farewell}'")
                farewell_audio = self.loop.run_until_complete(self.agent.text_to_speech(farewell))
                if farewell_audio:
                    self.utils.send_audio_chunks(self.conn, farewell_audio)
                self.conn.close()
                return
            
            roll_dice_triggers = ["roll dice", "roll the dice", "roll a die"]

            if any(fucntion in transcription.lower() for fucntion in roll_dice_triggers):
                print("[Function] Roll dice Trigger")
                self.functions.roll_dice()
                print("[Function] Roll dice done")

            animal_triggers = ["random animal", "animal name", "pick an animal"]

            if any(fucntion in transcription.lower() for fucntion in animal_triggers):
                print("[Function] Random Animal Trigger")
                self.functions.print_random_animal_name()
                print("[Function] Random Animal done")
                

            # Generate AI response
            print("🧠 [PIPELINE] Generating AI response...")
            ai_response = self.loop.run_until_complete(
                self.agent.get_ai_response(transcription, self.conversation_history)
            )
            print(f"🤖 [AI] Response: \"{ai_response}\"\n")

            # Convert to speech and send
            print("🔊 [PIPELINE] Converting response to speech...")
            response_audio = self.loop.run_until_complete(self.agent.text_to_speech(ai_response))
            if response_audio:
                print("📤 [PIPELINE] Sending audio response...")
                self.utils.send_audio_chunks(self.conn, response_audio)
                print("✅ [PIPELINE] Response sent! Listening for next input...\n")
            else:
                print("❌ [PIPELINE] Failed to generate response audio")

        except Exception as e:
            print(f"❌ [PIPELINE] Processing error: {e}")
            import traceback
            traceback.print_exc()

    def handle_client(self):
        """Handle AudioSocket client connection"""
        print(f"\n{'='*60}")
        print(f"🔗 [CONNECTION] New AudioSocket connection from {self.addr}")
        print(f"{'='*60}\n")

        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("⚡ [CONNECTION] TCP_NODELAY enabled for low latency")

        try:
            # Read UUID
            print("🆔 [SETUP] Waiting for call UUID...")
            msg_type, payload = self.utils.read_message(self.conn)
            if msg_type == MSG_UUID and len(payload) == 16:
                self.call_uuid = payload.hex()
                print(f"✅ [UUID] Call UUID received: {self.call_uuid}\n")
            else:
                print(f"❌ [SETUP] Expected UUID, got message type {msg_type}")
                return

            # Send greeting
            greeting = "Hello! I'm your AI assistant. How can I help you today?"
            print(f"👋 [GREETING] Sending: '{greeting}'")

            greeting_audio = self.loop.run_until_complete(self.agent.text_to_speech(greeting))
            if greeting_audio:
                print("📤 [GREETING] Audio generated successfully, sending to client...")
                self.utils.send_audio_chunks(self.conn, greeting_audio)
                print("✅ [GREETING] Greeting sent successfully\n")
            else:
                print("❌ [GREETING] Failed to generate greeting audio")
                return

            self.conversation_history.append({
                "role": "assistant",
                "content": greeting
            })

            print("🎧 [READY] Ready to receive audio from client...\n")

            # Main conversation loop
            audio_chunk_count = 0
            total_audio_received = 0

            while True:
                try:
                    msg_type, payload = self.utils.read_message(self.conn)
                    if msg_type is None:
                        print("🔌 [CONNECTION] Connection closed (read returned None)")
                        break

                    if msg_type == MSG_HANGUP:
                        print("📞 [HANGUP] Received hangup signal")
                        break

                    elif msg_type == MSG_AUDIO:
                        audio_chunk_count += 1
                        total_audio_received += len(payload)

                        if self.is_processing and self.vad.is_speaking:
                            print("🔄 [INTERRUPT] User started speaking during processing - Will process combined audio")
                            continue

                        # Add audio to VAD and check if speech is complete
                        should_process = self.vad.add_audio(payload)

                        if should_process:
                            if self.is_processing:
                                print("🚫 [INTERRUPT] Cancelling previous processing for combined audio")
                                self.processing_cancelled = True

                            self.is_processing = True

                            # Get the accumulated audio buffer
                            audio_data = self.vad.get_audio_buffer()

                            # Process in a separate thread to avoid blocking
                            processing_thread = threading.Thread(
                                target=self._process_audio_thread,
                                args=(audio_data,),
                                daemon=True
                            )
                            processing_thread.start()

                    elif msg_type == MSG_DTMF:
                        dtmf_digit = chr(payload[0]) if payload else ""
                        print(f"📞 [DTMF] Received DTMF digit: '{dtmf_digit}'")
                        if dtmf_digit == '*':
                            print("⭐ [DTMF] Star key pressed, ending call...")
                            farewell = "Goodbye! Have a great day!"
                            farewell_audio = self.loop.run_until_complete(self.agent.text_to_speech(farewell))
                            if farewell_audio:
                                self.utils.send_audio_chunks(self.conn, farewell_audio)
                            break

                    elif msg_type == MSG_ERROR:
                        error_code = payload[0] if payload else 0
                        print(f"🚨 [ERROR] AudioSocket error code: {error_code}")
                        break

                    else:
                        if DEBUG:
                            print(f"❓ [UNKNOWN] Unknown message type: {msg_type}")

                except Exception as e:
                    print(f"❌ [LOOP] Error in main loop: {e}")
                    continue

        except Exception as e:
            print(f"💥 [HANDLER] Client handler error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            print(f"\n{'='*60}")
            print(f"🔌 [DISCONNECT] Connection closed from {self.addr}")
            print(f"📊 [STATS] Total audio chunks received: {audio_chunk_count}")
            print(f"📊 [STATS] Total audio bytes received: {total_audio_received}")
            print(f"{'='*60}\n")

            self.loop.close()
            self.conn.close()

    def _process_audio_thread(self, audio_data):
        """Process audio in separate thread"""
        try:
            self.process_transcription(audio_data)
        finally:
            self.is_processing = False


def main():
    print(f"\n🚀 Starting Simple VAD AudioSocket Server")
    print(f"🌐 Host: {HOST}")
    print(f"🔌 Port: {PORT}")
    print(f"{'='*60}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print("🔧 Socket created")
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print("⚙️ Socket options configured (SO_REUSEADDR)")
        s.bind((HOST, PORT))
        print(f"🔗 Socket bound to {HOST}:{PORT}")
        s.listen()
        print("👂 Socket listening for connections")
        print(f"\n✅ Server ready! Waiting for Asterisk connections...\n")

        connection_count = 0

        while True:
            try:
                conn, addr = s.accept()
                connection_count += 1
                print(f"\n🎉 [SERVER] New connection #{connection_count} from {addr}")

                # Create handler instance for this connection
                handler = SimpleHandler(conn, addr)

                # Create new handler thread
                handler_thread = threading.Thread(
                    target=handler.handle_client, 
                    daemon=True,
                    name=f"Handler-{connection_count}"
                )
                handler_thread.start()
                print(f"🧵 [SERVER] Started handler thread: {handler_thread.name}")

            except KeyboardInterrupt:
                print("\n⛔ [SERVER] Keyboard interrupt received")
                print("🛑 [SERVER] Shutting down gracefully...")
                break
            except Exception as e:
                print(f"💥 [SERVER] Server error: {e}")
                import traceback
                traceback.print_exc()  


if __name__ == "__main__":
    main()
