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
import time
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
speech_file_path = Path(__file__).parent / "speech.mp3"


# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech
tts_engine.setProperty('volume', 0.9)  # Volume level

HOST = "0.0.0.0"
PORT = 9092

# AudioSocket message types
MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10
MSG_ERROR = 0xff

def text_to_audio_data(text):
    """
    Convert text to 16-bit PCM audio data at 8kHz mono (AudioSocket format)
    """
    if not text.strip():
        return b""
    
    # Create temporary file for TTS output
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_filename = temp_file.name
    
    try:
        # Generate speech and save to temp file
        tts_engine.save_to_file(text, temp_filename)
        tts_engine.runAndWait()
        
        # Read the generated audio file
        audio_data, sample_rate = sf.read(temp_filename)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 8kHz if needed
        if sample_rate != 8000:
            # Simple resampling (for production, use librosa.resample)
            step = sample_rate / 8000
            indices = np.arange(0, len(audio_data), step).astype(int)
            indices = indices[indices < len(audio_data)]
            audio_data = audio_data[indices]
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Convert to bytes
        return audio_data.tobytes()
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def send_audio_chunks(conn, audio_data):
    """
    Send audio data in 320-byte chunks (20ms of 16-bit 8kHz PCM)
    """
    chunk_size = 320  # 20ms of audio at 8kHz 16-bit mono
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        
        # Pad last chunk if necessary
        if len(chunk) < chunk_size:
            chunk = chunk + b'\x00' * (chunk_size - len(chunk))
        
        # Create AudioSocket message
        msg_type = MSG_AUDIO
        length = len(chunk)
        header = struct.pack('B', msg_type) + struct.pack('>H', length)
        
        try:
            conn.sendall(header + chunk)
            # Small delay between chunks to simulate real-time audio
            time.sleep(0.02)  # 20ms delay
        except Exception as e:
            print(f"[!] Error sending audio chunk: {e}")
            break

def read_message(conn):
    """Read a complete AudioSocket message"""
    try:
        # Read 3-byte header: type (1 byte) + length (2 bytes big-endian)
        header = conn.recv(3)
        if len(header) < 3:
            return None, None
        
        msg_type = header[0]
        payload_length = struct.unpack('>H', header[1:3])[0]  # Big-endian uint16
        
        # Read payload
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

    try:
        # First message (should be UUID)
        msg_type, payload = read_message(conn)
        if msg_type == MSG_UUID and len(payload) == 16:
            call_uuid = payload.hex()
            print(f"[UUID] Call UUID: {call_uuid}")
            
            # Send initial greeting
            greeting = "Hello! I'm an AI assistant. Please speak after the beep."
            # audio_data = text_to_audio_data(greeting)
            # if audio_data:
                # send_audio_chunks(conn, audio_data)
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="coral",
                input=greeting,
                instructions="Speak in a cheerful and positive tone.",
            ) as response:
                response.stream_to_file(speech_file_path)
        else:
            print(f"[!] Expected UUID, got type {msg_type}")

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
                # Append raw PCM audio
                audio_buffer.write(payload)

                # Process every 5 seconds of audio
                if audio_buffer.tell() >= 8000 * 2 * 5:  # 5 seconds at 8kHz, 16-bit
                    pcm_data = audio_buffer.getvalue()

                    # Convert PCM -> WAV (so Whisper can handle it)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                        with wave.open(temp_wav, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(8000)
                            wf.writeframes(pcm_data)
                        temp_wav_name = temp_wav.name

                    try:
                        # Transcribe with OpenAI Whisper
                        with open(temp_wav_name, "rb") as audio_file:
                            transcription = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file
                            )

                        text = transcription.text.strip()
                        print(f"[+] Transcribed: '{text}'")

                        # Only process if there's meaningful speech
                        if text and len(text) > 3:
                            print("[+] Generating AI response...")
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are an AI assistant on a phone call. Keep responses brief, natural, and conversational. Speak as if you're talking to someone directly."},
                                    {"role": "user", "content": text}
                                ],
                                max_tokens=100,
                                temperature=0.7
                            )

                            reply = response.choices[0].message.content.strip()
                            print(f"[+] AI reply: '{reply}'")

                            # Convert response text to speech and send to caller
                            response_audio = text_to_audio_data(reply)
                            if response_audio:
                                send_audio_chunks(conn, response_audio)

                    except Exception as e:
                        print(f"[!] Transcription/LLM Error: {e}")

                    finally:
                        if os.path.exists(temp_wav_name):
                            os.remove(temp_wav_name)

                        # Clear buffer for next segment
                        audio_buffer.seek(0)
                        audio_buffer.truncate(0)

                

            elif msg_type == MSG_DTMF:
                dtmf_digit = chr(payload[0]) if payload else ""
                print(f"[DTMF] Received digit: {dtmf_digit}")
                
                # Respond to specific DTMF digits
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
        print(f"[-] Connection closed from {addr}, total bytes: {audio_buffer.tell()}")
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
        print(f"🎧 AudioSocket server listening on {HOST}:{PORT}")
        
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    main()