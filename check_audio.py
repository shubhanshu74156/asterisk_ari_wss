import audioop
import socket
import threading
import wave
import io

HOST = "0.0.0.0"
PORT = 9092

def handle_client(conn, addr):
    print(f"[+] New AudioSocket connection from {addr}")

    audio_buffer = io.BytesIO()
    bytes_received = 0

    while True:
        data = conn.recv(160)  # 160 bytes = 20ms at 8kHz
        if not data:
            break
        # decode ulaw to 16-bit PCM
        pcm_data = audioop.ulaw2lin(data, 2)
        rms = audioop.rms(pcm_data, 2)
        print(f"RMS: {rms}")
        audio_buffer.write(pcm_data)
        bytes_received += len(pcm_data)
        if bytes_received % (8000 * 2 * 5) == 0:
            print(f"Received ~{bytes_received / (8000 * 2):.1f}s of audio...")

    print(f"[-] Connection closed from {addr}, total bytes: {bytes_received}")

    # Save to WAV
    output_file = f"recorded_{addr[1]}.wav"
    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(audio_buffer.getvalue())

    print(f"[✓] Audio saved to {output_file}\n")

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"🎧 AudioSocket test server listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr)).start()

if __name__ == "__main__":
    main()
