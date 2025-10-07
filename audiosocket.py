import socket
import struct
from threading import Thread

class AudioSocketConnection:
    def __init__(self, conn, addr):
        self.conn = conn
        self.addr = addr
        self.connected = True
        self.uuid = None
        
    def read(self):
        """Read audio packet from Asterisk"""
        try:
            header = self.conn.recv(3)
            if len(header) != 3:
                self.connected = False
                return None
                
            msg_type = header[0]
            length = struct.unpack('>H', header[1:3])[0]
            
            if msg_type == 0x00:  # Hangup
                self.connected = False
                return None
            elif msg_type == 0x01:  # UUID
                self.uuid = self.conn.recv(length)
            elif msg_type == 0x10:  # Audio data
                audio_data = self.conn.recv(length)
                return audio_data
            elif msg_type == 0x03:  # DTMF
                dtmf = self.conn.recv(length)
                return None
                
        except Exception as e:
            self.connected = False
            return None
    
    def write(self, audio_data):
        """Send audio back to Asterisk (must be 16-bit, 8kHz, mono PCM)"""
        try:
            length = len(audio_data)
            header = struct.pack('>BH', 0x10, length)
            self.conn.sendall(header + audio_data)
        except:
            self.connected = False
    
    def hangup(self):
        """Send hangup signal"""
        self.conn.sendall(bytes([0x00, 0x00, 0x00]))
        self.connected = False

class AudioSocketServer:
    def __init__(self, host='0.0.0.0', port=9092):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        
    def listen(self):
        """Wait for connection from Asterisk"""
        self.sock.listen(1)
        print(f"AudioSocket server listening on {self.host}:{self.port}")
        conn, addr = self.sock.accept()
        print(f"Connection from {addr}")
        return AudioSocketConnection(conn, addr)
