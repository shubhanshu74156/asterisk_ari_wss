import socket
import struct
import wave
import io

MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10
MSG_ERROR = 0xff


class RTPReceiver:
    def __init__(self, host='0.0.0.0', port=10000):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s = self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((host, port))
        self.audio_buffer = io.BytesIO()
        
    def parse_rtp_packet(self, data):
        """Parse RTP packet and extract audio payload"""
        if len(data) < 12:
            return None
        
        # RTP Header (12 bytes minimum)
        version = (data[0] >> 6) & 0x03
        padding = (data[0] >> 5) & 0x01
        extension = (data[0] >> 4) & 0x01
        csrc_count = data[0] & 0x0F
        
        payload_type = data[1] & 0x7F
        sequence = struct.unpack('>H', data[2:4])[0]
        timestamp = struct.unpack('>I', data[4:8])[0]
        ssrc = struct.unpack('>I', data[8:12])[0]
        
        # Calculate header size
        header_size = 12 + (4 * csrc_count)
        
        if extension:
            ext_len = struct.unpack('>H', data[header_size+2:header_size+4])[0]
            header_size += 4 + (ext_len * 4)
        
        # Extract payload (audio data)
        payload = data[header_size:]
        
        return {
            'sequence': sequence,
            'timestamp': timestamp,
            'payload': payload
        }
    
    def listen(self):
        """Listen for RTP packets"""
        print(f"[RTP] Listening on {self.host}:{self.port}")
        
        packet_count = 0
        
        try:
            while True:
                data, addr = self.s.recvfrom(2048)
                
                packet = self.parse_rtp_packet(data)
                if packet:
                    # Write audio payload to buffer
                    self.audio_buffer.write(packet['payload'])
                    packet_count += 1
                    
                    if packet_count % 100 == 0:
                        duration = packet_count * 0.02  # 20ms per packet
                        print(f"[RTP] Received {packet_count} packets (~{duration:.1f}s)")
                    
                    # Here you would send to your AI STT/LLM/TTS pipeline
                    # audio_data = packet['payload']
                    # await process_audio(audio_data)
                    
        except KeyboardInterrupt:
            print("\n[RTP] Stopped")
            self.save_audio()
    
    def save_audio(self):
        """Save received audio to WAV file"""
        if self.audio_buffer.tell() > 0:
            with wave.open("rtp_received.wav", "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)  # 16kHz for slin16
                wf.writeframes(self.audio_buffer.getvalue())
            
            print(f"[RTP] ✓ Saved audio to rtp_received.wav")

if __name__ == "__main__":
    receiver = RTPReceiver(host='0.0.0.0', port=9092)
    receiver.listen()
