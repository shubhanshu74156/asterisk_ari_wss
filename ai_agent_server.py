
import socket
import threading
import wave
import io
import struct
import soundfile as sf
import numpy as np
import pyttsx3
from io import BytesIO
import tempfile
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from enum import Enum

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

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

# Conversation states
class ConversationState(Enum):
    LISTENING = 1      # Collecting 5-second audio chunk
    PROCESSING = 2     # STT + LLM processing
    SPEAKING = 3       # Sending TTS response
    WAITING = 4        # Brief pause before next chunk

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
        with wave.open(temp_filename, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()
            # Convert to numpy array
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample to 8kHz if needed
        if sample_rate != 8000:
            # Simple resampling
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
    if not audio_data:
        return 0

    chunk_size = 320  # 20ms of audio at 8kHz 16-bit mono
    chunks_sent = 0

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
            chunks_sent += 1
            # Small delay between chunks to simulate real-time audio
            time.sleep(0.02)  # 20ms delay
        except Exception as e:
            print(f"[!] Error sending audio chunk: {e}")
            break

    return chunks_sent

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

def process_audio_chunk(audio_data, chunk_number):
    """Process 5-second audio chunk: STT -> LLM -> return response text"""
    try:
        # Convert PCM -> WAV for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            with wave.open(temp_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(8000)
                wf.writeframes(audio_data)
            temp_wav_name = temp_wav.name

        # Transcribe with OpenAI Whisper
        print(f"[{chunk_number}] 🎤 Transcribing...")
        with open(temp_wav_name, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        text = transcription.text.strip()
        print(f"[{chunk_number}] 📝 Transcribed: '{text}'")

        # Only process if there's meaningful speech
        if text and len(text) > 3:
            print(f"[{chunk_number}] 🤖 Generating AI response...")
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
            print(f"[{chunk_number}] 💬 AI reply: '{reply}'")
            return reply
        else:
            print(f"[{chunk_number}] 🔇 No meaningful speech detected")
            return None

    except Exception as e:
        print(f"[{chunk_number}] ❌ Processing error: {e}")
        return None
    finally:
        if 'temp_wav_name' in locals() and os.path.exists(temp_wav_name):
            os.remove(temp_wav_name)

def handle_client(conn, addr):
    print(f"[+] New AudioSocket connection from {addr}")
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    audio_buffer = io.BytesIO()
    call_uuid = None
    conversation_active = True
    state = ConversationState.LISTENING
    chunk_number = 0

    # Audio collection settings
    CHUNK_DURATION = 5  # seconds
    BYTES_PER_SECOND = 8000 * 2  # 8kHz * 16-bit
    CHUNK_SIZE_BYTES = CHUNK_DURATION * BYTES_PER_SECOND

    try:
        # First message (should be UUID)
        msg_type, payload = read_message(conn)
        if msg_type == MSG_UUID and len(payload) == 16:
            call_uuid = payload.hex()
            print(f"[UUID] 📞 Call UUID: {call_uuid}")

            # Send initial greeting
            state = ConversationState.SPEAKING
            print("[0] 🗣️  Sending greeting...")
            greeting = "Hello! I'm an AI assistant. Please speak for about 5 seconds after the beep."
            audio_data = text_to_audio_data(greeting)
            if audio_data:
                send_audio_chunks(conn, audio_data)

            # Brief pause, then start listening
            time.sleep(0.5)
            state = ConversationState.LISTENING
            chunk_number = 1
            print(f"[{chunk_number}] 👂 Listening for 5-second audio chunk...")
        else:
            print(f"[!] Expected UUID, got type {msg_type}")
            return

        # Main conversation loop
        while conversation_active:
            msg_type, payload = read_message(conn)
            if msg_type is None:
                print("[-] Connection closed")
                break

            if msg_type == MSG_HANGUP:
                print("[HANGUP] 📞 Received hangup signal")
                break

            elif msg_type == MSG_AUDIO:
                # Only collect audio when in LISTENING state
                if state == ConversationState.LISTENING:
                    audio_buffer.write(payload)

                    # Check if we have 5 seconds of audio
                    if audio_buffer.tell() >= CHUNK_SIZE_BYTES:
                        print(f"[{chunk_number}] ✅ Got 5-second chunk ({audio_buffer.tell()} bytes)")

                        # Change state to processing
                        state = ConversationState.PROCESSING

                        # Get audio data and reset buffer
                        pcm_data = audio_buffer.getvalue()
                        audio_buffer.seek(0)
                        audio_buffer.truncate(0)

                        # Process in separate thread to avoid blocking message reading
                        def process_and_respond():
                            nonlocal state, chunk_number, conversation_active

                            # Process audio chunk
                            response_text = process_audio_chunk(pcm_data, chunk_number)

                            if response_text:
                                # Convert to speech and send
                                state = ConversationState.SPEAKING
                                print(f"[{chunk_number}] 🗣️  Speaking response...")

                                response_audio = text_to_audio_data(response_text)
                                if response_audio:
                                    chunks_sent = send_audio_chunks(conn, response_audio)
                                    print(f"[{chunk_number}] ✅ Sent {chunks_sent} audio chunks")

                                # Brief pause before next listening session
                                state = ConversationState.WAITING
                                time.sleep(0.5)

                            # Start listening for next chunk
                            if conversation_active:
                                state = ConversationState.LISTENING
                                chunk_number += 1
                                print(f"[{chunk_number}] 👂 Listening for next 5-second chunk...")

                        # Start processing in background
                        processing_thread = threading.Thread(target=process_and_respond)
                        processing_thread.daemon = True
                        processing_thread.start()

                else:
                    # Ignore audio when not in listening state
                    pass

            elif msg_type == MSG_DTMF:
                dtmf_digit = chr(payload[0]) if payload else ""
                print(f"[DTMF] 📱 Received digit: {dtmf_digit}")

                # Respond to specific DTMF digits
                if dtmf_digit == '*':
                    state = ConversationState.SPEAKING
                    response = "You pressed star. Thank you for calling! Goodbye!"
                    audio_data = text_to_audio_data(response)
                    if audio_data:
                        send_audio_chunks(conn, audio_data)
                    conversation_active = False

            elif msg_type == MSG_ERROR:
                print(f"[ERROR] ❌ Asterisk error code: {payload[0] if payload else 0}")
                break

    except Exception as e:
        print(f"[!] Error: {e}")

    finally:
        print(f"[-] 📞 Connection closed from {addr}")
        print(f"[-] 📊 Total chunks processed: {chunk_number}")

        # Save final audio buffer if any
        if audio_buffer.tell() > 0:
            output_file = f"final_chunk_{call_uuid or addr[1]}.wav"
            with wave.open(output_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(8000)
                wf.writeframes(audio_buffer.getvalue())
            print(f"[✓] Final audio saved to {output_file}")

        conn.close()

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"🎧 AudioSocket server listening on {HOST}:{PORT}")
        print(f"📋 Conversation flow: Listen (5s) -> Process -> Speak -> Repeat")

        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    main()
