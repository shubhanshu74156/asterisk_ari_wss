#!/usr/bin/env python3
import json
import requests
import websocket
from uuid import uuid4
import base64
import time

# ---------------- ARI SETTINGS ----------------
ARI_URL = "http://localhost:8088/ari"
USERNAME = "ai_agent"
PASSWORD = "mysecurepassword"
APP = "my_app"

OUTBOUND_NUMBER = "+917999791954"
TRUNK = "twilio_trunk"

# ---------------- HELPERS ----------------
def ari_post(path, params=None, data=None, json_body=None):
    url = f"{ARI_URL}{path}"
    return requests.post(url, auth=(USERNAME, PASSWORD), params=params, data=data, json=json_body)

def ari_get(path, stream=False):
    url = f"{ARI_URL}{path}"
    return requests.get(url, auth=(USERNAME, PASSWORD), stream=stream)

# ---------------- GLOBAL STATE ----------------
bridge_id = None
rec_name = None

def start_external_media(bridge_id):
    """
    Create an ExternalMedia channel that streams bridge audio to UDP port 4000
    """
    host = "127.0.0.1:4000"
    params = {
        "app": APP,
        "external_host": host,
        "format": "slin16",
        "direction": "send"
    }

    r = requests.post(f"{ARI_URL}/channels/externalMedia",
                      params=params,
                      auth=(USERNAME, PASSWORD))
    if r.status_code == 200:
        ext = r.json()
        ext_id = ext["id"]
        print("ExternalMedia channel created:", ext_id)

        # add to the bridge
        requests.post(f"{ARI_URL}/bridges/{bridge_id}/addChannel",
                      params={"channel": ext_id},
                      auth=(USERNAME, PASSWORD))
        print("ExternalMedia added to bridge")
        return ext_id
    else:
        print("Failed to create ExternalMedia:", r.status_code, r.text)

# ---------------- WEBSOCKET CALLBACKS ----------------
def on_message(ws, message):
    global bridge_id, rec_name
    event = json.loads(message)
    etype = event.get("type")

    if etype == "StasisStart":
        channel = event["channel"]
        channel_id = channel["id"]
        print(f"[ARI] Channel entered Stasis: {channel_id}")

        # 1️⃣ Create a new bridge
        r = ari_post("/bridges", json_body={"type": "mixing"})
        bridge_id = r.json()["id"]
        print(f"[ARI] Bridge created: {bridge_id}")

        # 2️⃣ Add channel to bridge
        ari_post(f"/bridges/{bridge_id}/addChannel", params={"channel": channel_id})
        print(f"[ARI] Channel added to bridge")

        # 3️⃣ Start recording
        rec_name = f"rec_{uuid4().hex}"
        ari_post(f"/bridges/{bridge_id}/record", params={
            "name": rec_name,
            "format": "wav",
            "ifExists": "overwrite"
        })
        print(f"[ARI] Recording started: {rec_name}")

    elif etype == "StasisEnd":
        print(f"[ARI] Channel left Stasis — stopping recording")
        if rec_name:
            ari_post(f"/recordings/live/{rec_name}/stop")
            time.sleep(1)
            download_recording(rec_name)

    elif etype == "ChannelDestroyed":
        print(f"[ARI] Channel destroyed: {event['channel']['id']}")

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, code, msg):
    print("WebSocket closed:", code, msg)

def on_open(ws):
    print("[ARI] Connected to ARI WebSocket")
    originate_outbound_call()

# ---------------- ORIGINATE ----------------
def originate_outbound_call():
    endpoint = f"PJSIP/{OUTBOUND_NUMBER}@{TRUNK}"
    print(f"[ARI] Originating outbound call to {endpoint}")
    r = ari_post("/channels", data={
        "endpoint": endpoint,
        "app": APP,
        "callerId": "+15513654158"
    })
    if r.status_code < 300:
        print("[ARI] Outbound call originated successfully:", r.json()["id"])
    else:
        print("[ARI] Originate failed:", r.status_code, r.text)

# ---------------- DOWNLOAD RECORDING ----------------
def download_recording(name):
    print(f"[ARI] Downloading recording: {name}")
    r = ari_get(f"/recordings/stored/{name}/file", stream=True)
    if r.status_code == 200:
        with open(f"{name}.wav", "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        print(f"[ARI] Recording saved as {name}.wav")
    else:
        print("Failed to fetch recording:", r.status_code, r.text)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    ws_url = f"ws://localhost:8088/ari/events?app={APP}"
    auth_bytes = f"{USERNAME}:{PASSWORD}".encode("utf-8")
    auth_b64 = base64.b64encode(auth_bytes).decode("utf-8")
    headers = [f"Authorization: Basic {auth_b64}"]

    print("Starting ARI Stasis app listener...")
    ws = websocket.WebSocketApp(
        ws_url,
        header=headers,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    ws.run_forever()
