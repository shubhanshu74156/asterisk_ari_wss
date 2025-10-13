import socket
import threading
import wave
import io
import struct
import soundfile as sf
import whisper
import numpy as np
import pyttsx3
from io import BytesIO
import tempfile
import os
import time
from ollama import chat
from ollama import ChatResponse

model = whisper.load_model("base")

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 0.9)

HOST = "0.0.0.0"
PORT = 9092

# AudioSocket message types
MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10
MSG_ERROR = 0xff

# Speech detection configuration
SILENCE_THRESHOLD = 0.01  # RMS energy threshold for silence detection
FRAME_DURATION = 0.02  # 20ms frames
SAMPLE_RATE = 8000  # AudioSocket uses 8kHz
WAIT_TIME_AFTER_SILENCE = 3.0  # Wait 3 seconds after detecting silence

class SimpleSpeechDetector:
    def __init__(self):
        self.silence_start_time = None
        self.is_waiting = False
        self.speech_detected = False
        
    def calculate_rms_energy(self, audio_data):
        """Calculate RMS energy of audio data"""
        if len(audio_data) == 0:
            return 0
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        return np.sqrt(np.mean(audio_np ** 2)) / 32768.0  # Normalize to 0-1
    
    def process_audio_chunk(self, audio_chunk):
        """
        Process audio chunk and determine if we should trigger transcription
        Returns: True if should transcribe, False otherwise
        """
        current_time = time.time()
        energy = self.calculate_rms_energy(audio_chunk)
        
        if energy > SILENCE_THRESHOLD:
            # Speech detected
            self.speech_detected = True
            self.silence_start_time = None
            self.is_waiting = False
            return False
        else:
            # Silence detected
            if self.speech_detected and not self.is_waiting:
                # First time detecting silence after speech
                self.silence_start_time = current_time
                self.is_waiting = True
                print(f"[SILENCE] Started waiting 3 seconds...")
                return False
            elif self.is_waiting and self.silence_start_time:
                # Check if 3 seconds have passed
                elapsed = current_time - self.silence_start_time
                if elapsed >= WAIT_TIME_AFTER_SILENCE:
                    print(f"[TRIGGER] 3 seconds elapsed, processing speech...")
                    self.reset()
                    return True
            return False
    
    def reset(self):
        """Reset detector state"""
        self.silence_start_time = None
        self.is_waiting = False
        self.speech_detected = False

def text_to_audio_data(text):
    """Convert text to 16-bit PCM audio data at 8kHz mono"""
    if not text.strip():
        return b""
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_filename = temp_file.name
    
    try:
        tts_engine.save_to_file(text, temp_filename)
        tts_engine.runAndWait()
        
        audio_data, sample_rate = sf.read(temp_filename)
        
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        if sample_rate != 8000:
            step = sample_rate / 8000
            indices = np.arange(0, len(audio_data), step).astype(int)
            indices = indices[indices < len(audio_data)]
            audio_data = audio_data[indices]
        
        audio_data = (audio_data * 32767).astype(np.int16)
        return audio_data.tobytes()
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def send_audio_chunks(conn, audio_data):
    """Send audio data in 320-byte chunks"""
    chunk_size = 320
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        
        if len(chunk) < chunk_size:
            chunk = chunk + b'\x00' * (chunk_size - len(chunk))
        
        msg_type = MSG_AUDIO
        length = len(chunk)
        header = struct.pack('B', msg_type) + struct.pack('>H', length)
        
        try:
            conn.sendall(header + chunk)
            time.sleep(0.02)
        except Exception as e:
            print(f"[!] Error sending audio chunk: {e}")
            break

def read_message(conn):
    """Read a complete AudioSocket message"""
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

def handle_client(conn, addr):
    print(f"[+] New AudioSocket connection from {addr}")
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    audio_buffer = io.BytesIO()
    call_uuid = None
    conversation_active = True
    speech_detector = SimpleSpeechDetector()
    
    try:
        # First message (should be UUID)
        msg_type, payload = read_message(conn)
        if msg_type == MSG_UUID and len(payload) == 16:
            call_uuid = payload.hex()
            print(f"[UUID] Call UUID: {call_uuid}")
            
            # Send initial greeting
            greeting = "Hello! I'm an AI assistant. Please speak and I'll respond after a pause."
            audio_data = text_to_audio_data(greeting)
            if audio_data:
                send_audio_chunks(conn, audio_data)
        else:
            print(f"[!] Expected UUID, got type {msg_type if msg_type else 'None'}")

        # Process subsequent messages
        while conversation_active:
            msg_type, payload = read_message(conn)
            if msg_type is None:
                print("[-] Connection closed")
                break

            if msg_type == MSG_HANGUP:
                print("[HANGUP] Received hangup signal")
                break

            elif msg_type == MSG_AUDIO:
                # Always append to buffer (continue collecting)
                audio_buffer.write(payload)
                
                # Check if we should trigger transcription
                should_transcribe = speech_detector.process_audio_chunk(payload)
                
                if should_transcribe and audio_buffer.tell() > 0:
                    # Process the accumulated audio
                    pcm_data = audio_buffer.getvalue()
                    audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Transcribe the audio
                    result = model.transcribe(audio_np, fp16=False)
                    transcribed_text = result["text"].strip()
                    
                    print(f"[TRANSCRIPTION] User said: {transcribed_text}")
                    
                    if transcribed_text:
                        # Generate LLM response
                        response: ChatResponse = chat(model='gemma:2b', messages=[
                            {
                                'role': 'user',
                                'content': transcribed_text,
                            },
                        ])
                        
                        llm_reply = response.message.content
                        print(f"[AI RESPONSE] {llm_reply}")
                        
                        # Convert response to speech and send back
                        response_audio = text_to_audio_data(llm_reply)
                        if response_audio:
                            send_audio_chunks(conn, response_audio)
                    
                    # Clear the buffer for next turn
                    audio_buffer.seek(0)
                    audio_buffer.truncate(0)
                    print("[READY] Listening for next turn...")

            elif msg_type == MSG_DTMF:
                dtmf_digit = chr(payload[0]) if payload else ""
                print(f"[DTMF] Received digit: {dtmf_digit}")
                
                if dtmf_digit == '*':
                    response = "You pressed star. Goodbye!"
                    audio_data = text_to_audio_data(response)
                    if audio_data:
                        send_audio_chunks(conn, audio_data)
                    conversation_active = False

            elif msg_type == MSG_ERROR:
                print(f"[ERROR] Asterisk error code: {payload[0] if payload else 0}")
                break

    except Exception as e:
        print(f"[!] Error: {e}")

    finally:
        print(f"[-] Connection closed from {addr}")
        if audio_buffer.tell() > 0:
            output_file = f"recorded_{call_uuid or addr[1]}.wav"
            with wave.open(output_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(8000)
                wf.writeframes(audio_buffer.getvalue())
            print(f"[✓] Audio saved to {output_file}")
        conn.close()

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"🎧 Simple Turn-based AudioSocket server listening on {HOST}:{PORT}")
        
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    main()