import socket
import threading
import wave
import io
import struct
import numpy as np
import asyncio
import os
import tempfile
import time
from collections import deque
from openai import OpenAI
from dotenv import load_dotenv
import torch

# Load Silero VAD model
DEVICE = torch.device("cpu")  # or "cuda" if available
VAD_MODEL, VAD_UTILS = torch.hub.load(repo_or_dir='snakers4/silero-vad', 
                                      model='silero_vad', 
                                      force_reload=False, 
                                      trust_repo=True)
(get_speech_timestamps, save_audio, read_audio, VAD_iterator, collect_chunks) = VAD_UTILS

# Load environment
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEBUG = True

HOST = "0.0.0.0"
PORT = 9092

# AudioSocket message types
MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10
MSG_ERROR = 0xff


class SileroVAD:
    
    def __init__(self):
        pass

    def silero_detect_speech(self, audio_bytes, sample_rate=8000):
        """
        Returns True if speech is detected in audio_bytes
        """
        try:
            # Convert bytes to float tensor
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
            
            speech_timestamps = get_speech_timestamps(audio_tensor, VAD_MODEL, sampling_rate=sample_rate)
            return len(speech_timestamps) > 0
        except Exception as e:
            print(f"[!] Silero VAD error: {e}")
            return False

class Helper:

    def __init__(self):
        pass

    def calculate_rms(self, audio_data):
        """Calculate RMS (Root Mean Square) of audio data"""
        if len(audio_data) == 0:
            return 0.0
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) == 0:
                return 0.0
            rms = float(np.sqrt(np.mean(audio_array.astype(np.float64)**2)))
            return rms if not np.isnan(rms) else 0.0
        except Exception as e:
            return 0.0

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
        
class Utils:
    def __init__(self):
        pass
        
    def send_audio_chunks(conn, audio_data):
        """Send audio data in 320-byte chunks"""
        chunk_size = 320
        total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
        
        if DEBUG:
            print(f"[SEND] Sending {len(audio_data)} bytes in {total_chunks} chunks")
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk += b'\x00' * (chunk_size - len(chunk))
            header = struct.pack('B', MSG_AUDIO) + struct.pack('>H', len(chunk))
            try:
                conn.sendall(header + chunk)
                time.sleep(0.02)
            except Exception as e:
                print(f"[!] Error sending audio chunk: {e}")
                break
        
        if DEBUG:
            print(f"[SEND] Audio transmission complete")

    def read_message(conn):
        """Read one full AudioSocket message"""
        try:
            header = conn.recv(3)
            if len(header) < 3:
                return None, None
            msg_type = header[0]
            payload_length = struct.unpack('>H', header[1:3])[0]
            payload = b''
            while len(payload) < payload_length:
                chunk = conn.recv(payload_length - len(payload))
                if not chunk:
                    break
                payload += chunk
            return msg_type, payload
        except Exception as e:
            print(f"[!] Error reading message: {e}")
            return None, None

class Agent:
    def __init__(self):
      self.helper = Helper()
   
    def transcribe_audio(self, audio_bytes, sample_rate=8000):
        temp_path = None
        try:
            normalized_audio = self.helper.normalize_audio(audio_bytes)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_path = temp_audio.name
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(normalized_audio)
            with open(temp_path, 'rb') as audio_file:
                transcript = asyncio.to_thread(
                    client.audio.transcriptions.create,
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            return transcript.text.strip()
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

    def get_ai_response(self, user_text, conversation_history):
        """Get AI response using OpenAI Chat API"""
        try:
            conversation_history.append({
                "role": "user",
                "content": user_text
            })
            
            response = asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o-mini",
                messages=conversation_history,
                temperature=0.7,
                max_tokens=150
            )
            
            ai_text = response.choices[0].message.content
            conversation_history.append({
                "role": "assistant",
                "content": ai_text
            })
            
            return ai_text
        except Exception as e:
            print(f"[!] Chat API error: {e}")
            import traceback
            traceback.print_exc()
            return "I'm sorry, I didn't catch that. Could you please repeat?"
        
    def text_to_speech(text):
        """Convert text to speech using OpenAI TTS"""
        try:
            response = asyncio.to_thread(
                client.audio.speech.create,
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="pcm",
                speed=1.0
            )
            
            # OpenAI TTS outputs 24kHz PCM, downsample to 8kHz
            audio_data = response.content
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Downsample from 24kHz to 8kHz
            downsample_factor = 3
            downsampled = audio_array[::downsample_factor]
            
            return downsampled.tobytes()
        except Exception as e:
            print(f"[!] TTS error: {e}")
            import traceback
            traceback.print_exc()
            return None
        
class Handler:
    def __init__(self):
        self.vad = SileroVAD()
        self.agent = Agent()
        self.utils = Utils()
        self.helper = Helper()
        self.conversation_history = [
            {
                "role": "system",
                "content": "You are a helpful AI phone assistant. Keep your responses brief, clear, and conversational. Use simple language suitable for phone conversations. Keep responses under 3 sentences."
            }
        ]
        
    def handle_client(self, conn, addr):
        print(f"\n{'='*60}")
        print(f"[+] New AudioSocket connection from {addr}")
        print(f"{'='*60}\n")
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        audio_buffer = bytearray()
        call_uuid = None
        is_processing = False
        speech_detected = False

        try:
            # Read UUID
            msg_type, payload = self.utils.read_message(conn)
            if msg_type == MSG_UUID and len(payload) == 16:
                call_uuid = payload.hex()
                print(f"[UUID] Call UUID: {call_uuid}\n")
            else:
                print(f"[!] Expected UUID, got type {msg_type}")
                return

            # Send greeting
            greeting = "Hello! I'm your AI assistant. How can I help you today?"
            print(f"[AI] 🤖 {greeting}\n")
            greeting_audio = self.agent.text_to_speech(greeting)
            if greeting_audio:
                self.utils.send_audio_chunks(conn, greeting_audio)
            else:
                print("[!] Failed to generate greeting audio")
                return
            
            self.conversation_history.append({
                "role": "assistant",
                "content": greeting
            })

            print(f"[CALIBRATING] 📊 Learning background noise levels...\n")

            # Main conversation loop
            audio_chunk_count = 0
            while True:
                msg_type, payload = self.utils.read_message(conn)
                if msg_type is None:
                    print("[-] Connection closed (read returned None)")
                    break

                if msg_type == MSG_HANGUP:
                    print("[HANGUP] Received hangup signal")
                    break

                elif msg_type == MSG_AUDIO:
                    audio_chunk_count += 1
                    
                    if is_processing:
                        if DEBUG and audio_chunk_count % 50 == 0:
                            print(f"[SKIP] Ignoring audio while processing (chunk {audio_chunk_count})")
                        continue
                    
                    audio_buffer.extend(payload)
                    
                    # Calculate current stats
                    duration = len(audio_buffer) / (8000 * 2)
                    
                    # Check if user has finished speaking (adaptive)
                    has_speech = self.vad.silero_detect_speech(bytes(audio_buffer))
                   
                    if not has_speech:
                        is_processing = True
                        pcm_data = bytes(audio_buffer)
                        print(f"[PROCESSING] 🔄 Processing {len(pcm_data)} bytes ({duration:.2f}s)\n")

                        try:
                            # Transcribe user speech
                            print("[TRANSCRIBE] Converting speech to text...")
                            user_text = self.agent.transcribe_audio(pcm_data)

                            
                            if user_text and len(user_text.strip()) > 0:
                                print(f"[USER] 👤 \"{user_text}\"\n")

                                if user_text in ['Bye-bye']:
                                    conn.close()
                                
                                # Get AI response
                                print("[AI] Generating response...")
                                ai_response = self.agent.get_ai_response(user_text, self.conversation_history)
                                print(f"[AI] 🤖 \"{ai_response}\"\n")
                                
                                # Convert to speech and send
                                print("[TTS] Converting text to speech...")
                                response_audio = self.agent.text_to_speech(ai_response)
                                if response_audio:
                                    self.utils.send_audio_chunks(conn, response_audio)
                                    print("[LISTENING] 🎤 Waiting for user to speak...\n")
                                else:
                                    print("[!] Failed to generate response audio")
                            else:
                                print("[!] No transcription received or empty text\n")
                                # Ask user to repeat
                                retry_msg = "I didn't catch that. Could you please speak a bit louder?"
                                print(f"[AI] 🤖 {retry_msg}\n")
                                retry_audio = asyncio.run(self.agent.text_to_speech(retry_msg))
                                if retry_audio:
                                    self.utils.send_audio_chunks(conn, retry_audio)
                        
                        except Exception as e:
                            print(f"[!] Processing error: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            error_msg = "I'm sorry, I had trouble processing that. Please try again."
                            error_audio = asyncio.run(self.agent.text_to_speech(error_msg))
                            if error_audio:
                                self.utils.send_audio_chunks(conn, error_audio)
                        
                        finally:
                            # Clear buffer and reset
                            audio_buffer.clear()
                            audio_chunk_count = 0
                            speech_detected = False
                            is_processing = False

                elif msg_type == MSG_DTMF:
                    dtmf_digit = chr(payload[0]) if payload else ""
                    print(f"\n[DTMF] Received: {dtmf_digit}")
                    if dtmf_digit == '*':
                        farewell = "Goodbye! Have a great day!"
                        print(f"[AI] 🤖 {farewell}")
                        farewell_audio = asyncio.run(self.agent.text_to_speech(farewell))
                        if farewell_audio:
                            self.utils.send_audio_chunks(conn, farewell_audio)
                        break

                elif msg_type == MSG_ERROR:
                    error_code = payload[0] if payload else 0
                    print(f"[ERROR] AudioSocket error code: {error_code}")
                    break
                
                else:
                    print(f"[?] Unknown message type: {msg_type}")

        except Exception as e:
            print(f"\n[!] Client handler error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            print(f"\n{'='*60}")
            print(f"[-] Connection closed from {addr}")
            print(f"{'='*60}\n")
            conn.close()



def main():
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        
        while True:
            try:
                conn, addr = s.accept()
                threading.Thread(
                    target=Handler().handle_client, 
                    args=(conn, addr), 
                    daemon=True
                ).start()
            except KeyboardInterrupt:
                print("\n[!] Server shutting down...")
                break
            except Exception as e:
                print(f"[!] Server error: {e}")
                import traceback
                traceback.print_exc()  

if __name__ == "__main__":
    main()
