import asyncio
import aiohttp
import json
from typing import Dict


class ARICallManager:
    def __init__(self, ari_url="http://localhost:8088", 
                 ari_user="ai_agent", 
                 ari_password="mysecurepassword"):
        self.ari_url = ari_url
        self.ari_user = ari_user
        self.ari_password = ari_password
        self.auth = aiohttp.BasicAuth(ari_user, ari_password)
        self.app_name = "ai-calling-app"
        self.websocket = None
        self.session = None
        self.active_calls: Dict[str, dict] = {}
        
    async def connect_websocket(self):
        """Connect to ARI WebSocket for events"""
        self.session = aiohttp.ClientSession()
        
        ws_url = f"{self.ari_url.replace('http', 'ws')}/ari/events"
        params = {
            "app": self.app_name,
            "api_key": f"{self.ari_user}:{self.ari_password}"
        }
        
        try:
            # Use ClientWSTimeout for ws_close instead of deprecated timeout
            from aiohttp import ClientWSTimeout
            ws_timeout = ClientWSTimeout(ws_close=30)
            
            self.websocket = await self.session.ws_connect(
                ws_url,
                params=params,
                auth=self.auth,
                heartbeat=30,
                timeout=ws_timeout
            )
            
            print(f"[ARI] ✓ Connected to WebSocket: {self.app_name}")
            return self.websocket
            
        except Exception as e:
            print(f"[ERROR] WebSocket connection failed: {e}")
            raise
    
    async def originate_call(self, to_number: str, from_number: str):
        """Originate outbound call via ARI REST API"""
        
        url = f"{self.ari_url}/ari/channels"
        
        data = {
            "endpoint": f"PJSIP/{to_number}@twilio_trunk",
            "app": self.app_name,
            "callerId": from_number,
            "timeout": 30,
            "variables": {
                "AI_CALL": "true",
                "TARGET_NUMBER": to_number
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, auth=self.auth, json=data) as response:
                if response.status == 200:
                    channel_data = await response.json()
                    channel_id = channel_data['id']
                    
                    print(f"[CALL] ✓ Originated call to {to_number}")
                    print(f"[CHANNEL] ID: {channel_id}")
                    
                    self.active_calls[channel_id] = {
                        "to": to_number,
                        "from": from_number,
                        "state": "ringing",
                        "bridge_id": None,
                        "media_channel_id": None
                    }
                    
                    return channel_id
                else:
                    error = await response.text()
                    print(f"[ERROR] Failed to originate: {error}")
                    return None
    
    async def create_external_media_channel(self, rtp_host="127.0.0.1:9092"):
        """Create external media channel for audio streaming"""
        
        url = f"{self.ari_url}/ari/channels/externalMedia"
        
        params = {
            "app": self.app_name,
            "external_host": rtp_host,  # Your RTP receiver address
            "format": "slin16",  # 16-bit linear PCM, 16kHz
            "encapsulation": "rtp",
            "transport": "udp",
            "direction": "both"  # Send and receive audio
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, auth=self.auth, params=params) as response:
                if response.status == 200:
                    channel_data = await response.json()
                    media_channel_id = channel_data['id']
                    print(f"[EXTERNAL MEDIA] ✓ Created channel: {media_channel_id}")
                    return media_channel_id
                else:
                    error = await response.text()
                    print(f"[ERROR] External media failed: {error}")
                    return None
    
    async def create_bridge(self, bridge_type="mixing"):
        """Create a bridge to connect channels"""
        
        url = f"{self.ari_url}/ari/bridges"
        
        params = {"type": bridge_type}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, auth=self.auth, params=params) as response:
                if response.status == 200:
                    bridge_data = await response.json()
                    bridge_id = bridge_data['id']
                    print(f"[BRIDGE] ✓ Created bridge: {bridge_id}")
                    return bridge_id
                else:
                    error = await response.text()
                    print(f"[ERROR] Bridge creation failed: {error}")
                    return None
    
    async def add_channel_to_bridge(self, bridge_id: str, channel_id: str):
        """Add a channel to a bridge"""
        
        url = f"{self.ari_url}/ari/bridges/{bridge_id}/addChannel"
        
        params = {"channel": channel_id}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, auth=self.auth, params=params) as response:
                if response.status == 204:
                    print(f"[BRIDGE] ✓ Added channel {channel_id} to bridge {bridge_id}")
                    return True
                else:
                    error = await response.text()
                    print(f"[ERROR] Failed to add channel: {error}")
                    return False
    
    async def setup_audio_bridge(self, call_channel_id: str):
        """Set up bridge with external media for AI audio processing"""
        
        # Step 1: Create external media channel
        media_channel_id = await self.create_external_media_channel()
        if not media_channel_id:
            return False
        
        # Step 2: Create mixing bridge
        bridge_id = await self.create_bridge()
        if not bridge_id:
            return False
        
        # Step 3: Add both channels to bridge
        await self.add_channel_to_bridge(bridge_id, call_channel_id)
        await self.add_channel_to_bridge(bridge_id, media_channel_id)
        
        # Update call info
        if call_channel_id in self.active_calls:
            self.active_calls[call_channel_id]['bridge_id'] = bridge_id
            self.active_calls[call_channel_id]['media_channel_id'] = media_channel_id
        
        print(f"[AUDIO BRIDGE] ✓ Connected call to external media via bridge")
        return True
    
    async def handle_events(self):
        """Listen for ARI events via WebSocket"""
        
        try:
            async for msg in self.websocket:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(msg.data)
                    event_type = event.get('type')
                    
                    print(f"[EVENT] {event_type}")
                    
                    if event_type == 'StasisStart':
                        await self.on_stasis_start(event)
                        
                    elif event_type == 'ChannelStateChange':
                        await self.on_channel_state_change(event)
                        
                    elif event_type == 'StasisEnd':
                        await self.on_stasis_end(event)
                    
                    elif event_type == 'BridgeCreated':
                        print(f"[BRIDGE] Created")
                    
                    elif event_type == 'ChannelEnteredBridge':
                        channel_id = event['channel']['id']
                        bridge_id = event['bridge']['id']
                        print(f"[BRIDGE] Channel {channel_id} entered bridge {bridge_id}")
                
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    print(f"[WS] Connection closed")
                    break
                    
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"[WS ERROR] {self.websocket.exception()}")
                    break
                    
        except asyncio.CancelledError:
            print("[WS] Event handler cancelled")
        finally:
            await self.cleanup()
    
    async def on_stasis_start(self, event):
        """Channel entered Stasis application"""
        channel = event['channel']
        channel_id = channel['id']
        state = channel['state']
        name = channel.get('name', '')
        
        print(f"[STASIS START] Channel {channel_id} ({name}) - State: {state}")
        
        # Only process the actual call channel, not the external media channel
        if channel_id in self.active_calls:
            await self.answer_channel(channel_id)
            await asyncio.sleep(0.5)  # Wait for answer
            
            # Set up audio bridge instead of continuing to dialplan
            await self.setup_audio_bridge(channel_id)
    
    async def on_channel_state_change(self, event):
        """Channel state changed"""
        channel = event['channel']
        channel_id = channel['id']
        state = channel['state']
        
        print(f"[STATE] Channel {channel_id}: {state}")
        
        if channel_id in self.active_calls:
            self.active_calls[channel_id]['state'] = state.lower()
    
    async def on_stasis_end(self, event):
        """Channel left Stasis (hung up)"""
        channel = event['channel']
        channel_id = channel['id']
        
        print(f"[STASIS END] Channel {channel_id} hung up")
        
        # Clean up bridge and media channel if this was a call channel
        if channel_id in self.active_calls:
            call_info = self.active_calls[channel_id]
            
            # Hangup media channel if exists
            if call_info.get('media_channel_id'):
                await self.hangup_channel(call_info['media_channel_id'])
            
            # Delete bridge if exists
            if call_info.get('bridge_id'):
                await self.delete_bridge(call_info['bridge_id'])
            
            del self.active_calls[channel_id]
    
    async def answer_channel(self, channel_id: str):
        """Answer a channel"""
        url = f"{self.ari_url}/ari/channels/{channel_id}/answer"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, auth=self.auth) as response:
                if response.status == 204:
                    print(f"[ANSWERED] ✓ Channel {channel_id}")
                    return True
                return False
    
    async def hangup_channel(self, channel_id: str):
        """Hangup a channel"""
        url = f"{self.ari_url}/ari/channels/{channel_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, auth=self.auth) as response:
                if response.status == 204:
                    print(f"[HANGUP] ✓ Channel {channel_id}")
                    return True
                return False
    
    async def delete_bridge(self, bridge_id: str):
        """Delete a bridge"""
        url = f"{self.ari_url}/ari/bridges/{bridge_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, auth=self.auth) as response:
                if response.status == 204:
                    print(f"[BRIDGE] ✓ Deleted bridge {bridge_id}")
                    return True
                return False
    
    async def cleanup(self):
        """Close WebSocket and session properly"""
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
        if self.session and not self.session.closed:
            await self.session.close()
        print("[CLEANUP] ✓ Closed all connections")


async def main():
    manager = ARICallManager(
        ari_url="http://localhost:8088",
        ari_user="ai_agent",
        ari_password="mysecurepassword"
    )
    
    try:
        await manager.connect_websocket()
        event_task = asyncio.create_task(manager.handle_events())
        await asyncio.sleep(1)
        
        channel_id = await manager.originate_call(
            to_number="+917999791954",
            from_number="+15513654158"
        )
        
        if channel_id:
            print(f"[SUCCESS] Call initiated: {channel_id}")
        
        await event_task
        
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Interrupted by user")
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        await manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
