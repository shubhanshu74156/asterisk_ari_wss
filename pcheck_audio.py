import socket
import threading
import wave
import io
import struct

HOST = "0.0.0.0"
PORT = 9092

# AudioSocket message types
MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10
MSG_ERROR = 0xff

def read_message(conn):
    """Read a complete AudioSocket message"""
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

def send_audio(conn, audio_data):
    """Send audio back to Asterisk via AudioSocket"""
    # AudioSocket expects 320 bytes (20ms of 16-bit 8kHz PCM)
    if len(audio_data) != 320:
        # Pad or trim to exact size
        if len(audio_data) < 320:
            audio_data = audio_data + b'\x00' * (320 - len(audio_data))
        else:
            audio_data = audio_data[:320]
    
    # Create message: type 0x10 + length (big-endian) + audio data
    msg_type = MSG_AUDIO
    length = len(audio_data)
    header = struct.pack('B', msg_type) + struct.pack('>H', length)
    conn.sendall(header + audio_data)

def handle_client(conn, addr):
    print(f"[+] New AudioSocket connection from {addr}")
    
    # Enable TCP_NODELAY for low latency
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    audio_buffer = io.BytesIO()
    call_uuid = None
    
    try:
        # First message should be UUID
        msg_type, payload = read_message(conn)
        
        if msg_type == MSG_UUID and len(payload) == 16:
            # Convert binary UUID to string representation
            call_uuid = payload.hex()
            print(f"[UUID] Call UUID: {call_uuid}")
        else:
            print(f"[!] Expected UUID, got type {msg_type}")
        
        # Process audio messages
        while True:
            msg_type, payload = read_message(conn)
            
            if msg_type is None:
                print("[-] Connection closed")
                break
            
            if msg_type == MSG_HANGUP:
                print("[HANGUP] Received hangup signal")
                break
            
            elif msg_type == MSG_AUDIO:
                # Audio is already in signed linear 16-bit PCM format
                audio_buffer.write(payload)
                
                # Echo audio back (for testing bidirectional flow)
                # Replace this with your AI-generated audio
                send_audio(conn, payload)
                
                if audio_buffer.tell() % (8000 * 2 * 5) == 0:
                    print(f"Received ~{audio_buffer.tell() / (8000 * 2):.1f}s of audio...")
            
            elif msg_type == MSG_DTMF:
                dtmf_digit = chr(payload[0])
                print(f"[DTMF] Received digit: {dtmf_digit}")
            
            elif msg_type == MSG_ERROR:
                error_code = payload[0] if payload else 0
                print(f"[ERROR] Asterisk error code: {error_code}")
                break
    
    except Exception as e:
        print(f"[!] Error: {e}")
    
    finally:
        print(f"[-] Connection closed from {addr}, total bytes: {audio_buffer.tell()}")
        
        # Save to WAV
        if audio_buffer.tell() > 0:
            output_file = f"recorded_{call_uuid or addr[1]}.wav"
            with wave.open(output_file, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(8000)  # 8kHz
                wf.writeframes(audio_buffer.getvalue())
            
            print(f"[✓] Audio saved to {output_file}\n")
        
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
