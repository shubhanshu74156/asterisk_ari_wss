import struct
import time

DEBUG = True

# AudioSocket message types
MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10
MSG_ERROR = 0xff

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