import socket
import threading
import subprocess
import io
import torch
import sounddevice as sd
import openai
import wave
from gtts import gTTS

# === CONFIG ===
HOST = "0.0.0.0"
PORT = 9092
openai.api_key = "YOUR_OPENAI_API_KEY"  # optional, if using GPT-4

# === AUDIO SOCKET HANDLER ===
def handle_client(conn, addr):
    print(f"New AudioSocket connection from {addr}")
    
    audio_buffer = io.BytesIO()
    
    while True:
        data = conn.recv(160)
        if not data:
            break
        audio_buffer.write(data)
    
    # Save received audio for STT
    with wave.open("call_input.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(audio_buffer.getvalue())
    
    # --- Step 1: Speech to Text (STT) ---
    print("Transcribing...")
    result = openai.Audio.transcriptions.create(
        model="whisper-1",
        file=open("call_input.wav", "rb")
    )
    text = result.text
    print(f"Caller said: {text}")
    
    # --- Step 2: LLM Response ---
    print("Generating LLM response...")
    response = openai.Chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful voice assistant."},
            {"role": "user", "content": text}
        ]
    )
    reply = response.choices[0].message.content
    print(f"AI reply: {reply}")
    
    # --- Step 3: Text to Speech (TTS) ---
    print("Converting to speech...")
    tts = gTTS(reply)
    tts.save("response.wav")

    # Send the TTS audio back to Asterisk
    with open("response.wav", "rb") as f:
        audio_data = f.read()
        conn.sendall(audio_data)

    conn.close()
    print("Connection closed.")


# === MAIN SERVER LOOP ===
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"AudioSocket server listening on {HOST}:{PORT}")
    
    while True:
        conn, addr = s.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
