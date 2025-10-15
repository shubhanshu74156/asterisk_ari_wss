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

# Load environment
load_dotenv()

HOST = "0.0.0.0"
PORT = 9092

# AudioSocket message types
MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10
MSG_ERROR = 0xff

# Voice Activity Detection parameters - ADJUSTED FOR LOW AUDIO LEVELS
VAD_THRESHOLD = 30  # Much lower threshold for quiet audio
SILENCE_DURATION = 2.0
MIN_SPEECH_DURATION = 0.5
MAX_AUDIO_DURATION = 30.0

# Adaptive VAD parameters
NOISE_FLOOR_SAMPLES = 50  # Number of chunks to calculate noise floor
SPEECH_MULTIPLIER = 2.5  # Speech must be this many times louder than noise

# Debug mode
DEBUG = True

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- start a background asyncio loop for thread-safe coroutine execution ----------
import concurrent.futures
import threading


_background_loop = None

def _start_background_loop():
    global _background_loop
    if _background_loop is None:
        _background_loop = asyncio.new_event_loop()
        def _run_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()
        t = threading.Thread(target=_run_loop, args=(_background_loop,), daemon=True)
        t.start()

def run_async(coro, timeout=None):
    """
    Schedule coroutine on the background loop and return result.
    Blocks current thread until result is available (or timeout).
    """
    _start_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, _background_loop)
    return future.result(timeout)  # may raise exceptions from the coroutine
# ----------------------------------------------------------------------------------------

# -----------------------------
#  ADAPTIVE VOICE ACTIVITY DETECTION
# -----------------------------
class AdaptiveVAD:
    def __init__(self):
        self.noise_samples = deque(maxlen=NOISE_FLOOR_SAMPLES)
        self.noise_floor = 10.0  # Initial estimate
        self.calibrated = False
        self.speech_threshold = max(self.noise_floor * SPEECH_MULTIPLIER, VAD_THRESHOLD)
    
    def update_noise_floor(self, rms):
        """Update noise floor estimate"""
        if not np.isnan(rms) and rms > 0:
            self.noise_samples.append(rms)
            if len(self.noise_samples) >= 20:
                # Use median to be robust against speech
                self.noise_floor = np.median(list(self.noise_samples))
                self.speech_threshold = max(self.noise_floor * SPEECH_MULTIPLIER, VAD_THRESHOLD)
                if not self.calibrated and len(self.noise_samples) >= NOISE_FLOOR_SAMPLES:
                    self.calibrated = True
                    print(f"[VAD] 🎯 Calibrated! Noise floor: {self.noise_floor:.1f}, Speech threshold: {self.speech_threshold:.1f}")
    
    def is_speech(self, rms):
        """Determine if RMS indicates speech"""
        return rms > self.speech_threshold

def calculate_rms(audio_data):
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

def normalize_audio(audio_data):
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

def detect_speech_end_adaptive(audio_buffer, vad, silence_duration=SILENCE_DURATION, sample_rate=8000):
    """
    Adaptive VAD - detects speech end based on learned noise floor
    Returns: (is_speech_ended, has_speech, current_rms)
    """
    if len(audio_buffer) == 0:
        return False, False, 0.0
    
    # Analyze last N seconds for silence
    bytes_per_sec = sample_rate * 2
    silence_bytes = int(silence_duration * bytes_per_sec)
    
    # Calculate current RMS (last 100ms)
    current_chunk_size = min(len(audio_buffer), 1600)
    current_rms = calculate_rms(audio_buffer[-current_chunk_size:])
    
    # Update noise floor with current sample
    vad.update_noise_floor(current_rms)
    
    if len(audio_buffer) < silence_bytes:
        # Check if current audio has speech
        has_speech = vad.is_speech(current_rms)
        return False, has_speech, current_rms
    
    # Check if last silence_duration seconds are silent
    recent_audio = audio_buffer[-silence_bytes:]
    recent_rms = calculate_rms(recent_audio)
    
    # Check earlier audio for speech
    if len(audio_buffer) > silence_bytes:
        earlier_audio = audio_buffer[:-silence_bytes]
        earlier_rms = calculate_rms(earlier_audio)
        has_speech = vad.is_speech(earlier_rms)
    else:
        has_speech = False
    
    is_silent = not vad.is_speech(recent_rms)
    
    return is_silent and has_speech, has_speech, current_rms

# -----------------------------
#  OPENAI VOICE PROCESSING
# -----------------------------
async def transcribe_audio(audio_data, sample_rate=8000):
    temp_path = None
    try:
        normalized_audio = normalize_audio(audio_data)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(normalized_audio)
        with open(temp_path, 'rb') as audio_file:
            transcript = await asyncio.to_thread(
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

async def get_ai_response(user_text, conversation_history):
    """Get AI response using OpenAI Chat API"""
    try:
        conversation_history.append({
            "role": "user",
            "content": user_text
        })
        
        response = await asyncio.to_thread(
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

async def text_to_speech(text):
    """Convert text to speech using OpenAI TTS"""
    try:
        response = await asyncio.to_thread(
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

# -----------------------------
#   AUDIO SOCKET UTILITIES
# -----------------------------
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

# -----------------------------
#   CLIENT HANDLER
# -----------------------------
def handle_client(conn, addr):
    print(f"\n{'='*60}")
    print(f"[+] New AudioSocket connection from {addr}")
    print(f"{'='*60}\n")
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    audio_buffer = bytearray()
    call_uuid = None
    is_processing = False
    vad = AdaptiveVAD()
    speech_detected = False
    conversation_history = [
        {
            "role": "system",
            "content": "You are a helpful AI phone assistant. Keep your responses brief, clear, and conversational. Use simple language suitable for phone conversations. Keep responses under 3 sentences."
        }
    ]

    try:
        # Read UUID
        msg_type, payload = read_message(conn)
        if msg_type == MSG_UUID and len(payload) == 16:
            call_uuid = payload.hex()
            print(f"[UUID] Call UUID: {call_uuid}\n")
        else:
            print(f"[!] Expected UUID, got type {msg_type}")
            return

        # Send greeting
        greeting = "Hello! I'm your AI assistant. How can I help you today?"
        print(f"[AI] 🤖 {greeting}\n")
        greeting_audio = run_async(text_to_speech(greeting))
        if greeting_audio:
            send_audio_chunks(conn, greeting_audio)
        else:
            print("[!] Failed to generate greeting audio")
            return
        
        conversation_history.append({
            "role": "assistant",
            "content": greeting
        })

        print(f"[CALIBRATING] 📊 Learning background noise levels...\n")

        # Main conversation loop
        audio_chunk_count = 0
        while True:
            msg_type, payload = read_message(conn)
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
                speech_ended, has_speech, current_rms = detect_speech_end_adaptive(
                    bytes(audio_buffer),
                    vad,
                    silence_duration=SILENCE_DURATION
                )
                
                # Track if we've detected any speech
                if has_speech and not speech_detected:
                    speech_detected = True
                    print(f"[VAD] 🎤 Speech detected! (RMS: {current_rms:.1f} > threshold: {vad.speech_threshold:.1f})\n")
                
                # Debug output every 50 chunks
                if DEBUG and audio_chunk_count % 50 == 0:
                    status = "SPEECH" if has_speech else "SILENCE"
                    print(f"[AUDIO] {duration:.1f}s | RMS: {current_rms:.1f} | Threshold: {vad.speech_threshold:.1f} | {status}")
                
                # Force processing if max duration reached
                force_process = duration > MAX_AUDIO_DURATION and vad.calibrated
                
                if force_process:
                    print(f"\n[!] Max duration reached ({duration:.1f}s), forcing processing\n")
                
                # Process when speech ends or max duration
                should_process = (speech_ended and duration > MIN_SPEECH_DURATION) or force_process
                
                if should_process and vad.calibrated:
                    is_processing = True
                    pcm_data = bytes(audio_buffer)
                    print(f"[PROCESSING] 🔄 Processing {len(pcm_data)} bytes ({duration:.2f}s)\n")

                    try:
                        # Transcribe user speech
                        print("[TRANSCRIBE] Converting speech to text...")
                        user_text = run_async(transcribe_audio(pcm_data))

                        
                        if user_text and len(user_text.strip()) > 0:
                            print(f"[USER] 👤 \"{user_text}\"\n")

                            if user_text in ['Bye-bye']:
                                conn.close()
                            
                            # Get AI response
                            print("[AI] Generating response...")
                            ai_response = run_async(get_ai_response(user_text, conversation_history))
                            print(f"[AI] 🤖 \"{ai_response}\"\n")
                            
                            # Convert to speech and send
                            print("[TTS] Converting text to speech...")
                            response_audio = run_async(text_to_speech(ai_response))
                            if response_audio:
                                send_audio_chunks(conn, response_audio)
                                print("[LISTENING] 🎤 Waiting for user to speak...\n")
                            else:
                                print("[!] Failed to generate response audio")
                        else:
                            print("[!] No transcription received or empty text\n")
                            # Ask user to repeat
                            retry_msg = "I didn't catch that. Could you please speak a bit louder?"
                            print(f"[AI] 🤖 {retry_msg}\n")
                            retry_audio = asyncio.run(text_to_speech(retry_msg))
                            if retry_audio:
                                send_audio_chunks(conn, retry_audio)
                    
                    except Exception as e:
                        print(f"[!] Processing error: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        error_msg = "I'm sorry, I had trouble processing that. Please try again."
                        error_audio = asyncio.run(text_to_speech(error_msg))
                        if error_audio:
                            send_audio_chunks(conn, error_audio)
                    
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
                    farewell_audio = asyncio.run(text_to_speech(farewell))
                    if farewell_audio:
                        send_audio_chunks(conn, farewell_audio)
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

# -----------------------------
#   MAIN SERVER LOOP
# -----------------------------
def main():
    print("\n" + "=" * 60)
    print("🎧 AI PHONE AGENT SERVER (Adaptive VAD)")
    print("=" * 60)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Base VAD Threshold: {VAD_THRESHOLD} RMS")
    print(f"Speech Multiplier: {SPEECH_MULTIPLIER}x noise floor")
    print(f"Silence Duration: {SILENCE_DURATION}s")
    print(f"Min Speech Duration: {MIN_SPEECH_DURATION}s")
    print(f"Max Audio Duration: {MAX_AUDIO_DURATION}s")
    print("=" * 60)
    print("\n✅ Server ready! Waiting for connections...\n")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        
        while True:
            try:
                conn, addr = s.accept()
                threading.Thread(
                    target=handle_client, 
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