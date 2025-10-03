import requests
import websocket
import json
import base64

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

# ---------------- GLOBALS ----------------
channels = []   # Active call channels
bridge_id = None

# ---------------- ARI HELPERS ----------------
def create_bridge():
    global bridge_id
    url = f"{BASE_URL}/bridges"
    r = requests.post(url, auth=(USERNAME, PASSWORD))
    bridge_id = r.json()["id"]
    print("Bridge created:", bridge_id)
    return bridge_id

def add_to_bridge(channel_id):
    global bridge_id
    url = f"{BASE_URL}/bridges/{bridge_id}/addChannel"
    data = {"channel": channel_id}
    r = requests.post(url, auth=(USERNAME, PASSWORD), data=data)
    print(f"Added {channel_id} to bridge:", r.status_code)

def originate_call(number, callerid="+15513654158"):
    url = f"{BASE_URL}/channels"
    data = {
        "endpoint": f"PJSIP/{number}@twilio_trunk",
        "app": APP,
        "callerId": callerid
    }
    r = requests.post(url, auth=(USERNAME, PASSWORD), data=data)
    channel_id = r.json()["id"]
    print(f"Originate {number} -> channel {channel_id}")
    return channel_id

# ---------------- WEBSOCKET HANDLERS ----------------
def on_message(ws, message):
    global channels, bridge_id
    event = json.loads(message)
    etype = event["type"]

    if etype == "StasisStart":
        chan_id = event["channel"]["id"]
        print("StasisStart:", chan_id)
        channels.append(chan_id)

        # When 2 channels are ready, create bridge
        if len(channels) == 2 and bridge_id is None:
            create_bridge()
            add_to_bridge(channels[0])
            add_to_bridge(channels[1])
            print("Two calls are now connected!")

    elif etype == "ChannelDestroyed":
        print("Call ended:", event["channel"]["id"])
        if event["channel"]["id"] in channels:
            channels.remove(event["channel"]["id"])

def on_error(ws, error):
    print("WebSocket Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    print("Connected to ARI WebSocket")
    # Originate two outbound calls automatically
    originate_call("+917999791954")  # User 1
    originate_call("+918576964227")  # User 2

# ---------------- RUN ----------------
if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        WS_URL,
        header=headers,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    ws.run_forever()
