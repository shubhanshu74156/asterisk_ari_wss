import requests
import websocket
import json
import base64
import socket
import numpy as np
import whisper
from threading import Thread

# ---------------- ARI SETTINGS ----------------
USERNAME = "ai_agent"
PASSWORD = "mysecurepassword"
HOST = "localhost:8088"
APP = "my_app"

BASE_URL = f"http://{HOST}/ari"
WS_URL = f"ws://{HOST}/ari/events?app={APP}"

auth_string = f"{USERNAME}:{PASSWORD}"
auth_header = base64.b64encode(auth_string.encode()).decode()
headers = {"Authorization": "Basic " + auth_header}

# ---------------- WHISPER SETTINGS ----------------
model = whisper.load_model("small")  # choose tiny, base, small, medium, large
UDP_IP = "0.0.0.0"
UDP_PORT = 5001

# ---------------- GLOBALS ----------------
caller_channel = None
external_channel = None
bridge_id = None

# ---------------- REST HELPERS ----------------
def originate_call(number, callerid="+15513654158"):
    url = f"{BASE_URL}/channels"
    data = {
        "endpoint": f"PJSIP/{number}@twilio_trunk",
        "app": APP,
        "callerId": callerid
    }
    r = requests.post(url, auth=(USERNAME, PASSWORD), data=data)
    print("Originate:", r.status_code, r.text)
    return r.json()["id"]

def create_external_media():
    url = f"{BASE_URL}/channels/externalMedia"
    params = {
        "app": APP,
        "external_host": f"{UDP_IP}:{UDP_PORT}",
        "format": "slin16"
    }
    r = requests.post(url, auth=(USERNAME, PASSWORD), params=params)
    print("ExternalMedia:", r.status_code, r.text)
    return r.json()["id"]

def create_bridge():
    global bridge_id
    url = f"{BASE_URL}/bridges"
    r = requests.post(url, auth=(USERNAME, PASSWORD))
    bridge = r.json()
    bridge_id = bridge["id"]
    print("Bridge created:", bridge_id)
    return bridge_id

def add_to_bridge(channel_id):
    global bridge_id
    url = f"{BASE_URL}/bridges/{bridge_id}/addChannel"
    data = {"channel": channel_id}
    r = requests.post(url, auth=(USERNAME, PASSWORD), data=data)
    print(f"Added {channel_id} to bridge:", r.status_code)

# ---------------- WHISPER RTP LISTENER ----------------
def rtp_whisper_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for RTP on {UDP_IP}:{UDP_PORT} ...")

    buffer = bytearray()
    while True:
        data, addr = sock.recvfrom(2048)
        payload = data[12:]  # remove RTP header
        buffer.extend(payload)

        # Process audio every 1 second (16kHz, 16-bit PCM)
        if len(buffer) > 32000 * 2:  # 1 sec * 16000 samples * 2 bytes
            audio = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
            result = model.transcribe(audio, language="en")
            if result.get("text"):
                print("User said:", result["text"])
            buffer.clear()

# ---------------- WEBSOCKET HANDLERS ----------------
def on_message(ws, message):
    global caller_channel, external_channel
    event = json.loads(message)
    etype = event["type"]

    if etype == "StasisStart":
        chan_id = event["channel"]["id"]
        print("StasisStart:", chan_id)

        if not caller_channel:
            caller_channel = chan_id
            print("Caller channel:", caller_channel)

            external_channel = create_external_media()
            print("External media channel:", external_channel)

            create_bridge()
            add_to_bridge(caller_channel)
            add_to_bridge(external_channel)

    elif etype == "ChannelDestroyed":
        print("Call ended:", event["channel"]["id"])

def on_error(ws, error):
    print("WebSocket Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    print("Connected to ARI WebSocket")
    originate_call("+917999791954", callerid="+15513654158")
    # originate_call("+918576964227", callerid="+15513654158")

# ---------------- RUN ----------------
if __name__ == "__main__":
    # Start RTP listener in a separate thread
    Thread(target=rtp_whisper_listener, daemon=True).start()

    # Start ARI WebSocket
    ws = websocket.WebSocketApp(
        WS_URL,
        header=headers,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    ws.run_forever()
