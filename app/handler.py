import socket
import asyncio
from vad import SimpleVAD
from agent import Agent
from utilities import Utils
import threading

DEBUG = True


# AudioSocket message types
MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10
MSG_ERROR = 0xff


class SimpleHandler:
    def __init__(self, conn, addr):
        self.conn = conn
        self.addr = addr
        self.utils = Utils()
        self.agent = Agent()
        self.vad = SimpleVAD()
        self.conversation_history = [
            {
                "role": "system",
                "content": "You are a helpful AI phone assistant. Keep your responses brief, clear, and conversational. Use simple language suitable for phone conversations. Keep responses under 3 sentences."
            }
        ]
        self.is_processing = False
        self.call_uuid = None

        # Create event loop for async operations
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def process_transcription(self, audio_data):
        """Process the audio data and generate AI response"""
        try:
            print(f"🔄 [PIPELINE] Processing {len(audio_data)} bytes of audio...")

            # Transcribe audio
            transcription = self.loop.run_until_complete(
                self.agent.transcribe_audio(audio_data)
            )

            if not transcription or len(transcription.strip()) == 0:
                print("⚠️ [PIPELINE] Empty transcription received")
                return

            print(f"👤 [USER] Transcription: \"{transcription}\"\n")

            # Check for goodbye phrases
            goodbye_phrases = ['bye', 'goodbye', 'bye-bye', 'see you', 'farewell']
            if any(phrase in transcription.lower() for phrase in goodbye_phrases):
                print("👋 [GOODBYE] Goodbye detected, ending conversation...")
                farewell = "Goodbye! Have a great day!"
                print(f"🤖 [AI] Final response: '{farewell}'")
                farewell_audio = self.loop.run_until_complete(self.agent.text_to_speech(farewell))
                if farewell_audio:
                    self.utils.send_audio_chunks(self.conn, farewell_audio)
                self.conn.close()
                return

            # Generate AI response
            print("🧠 [PIPELINE] Generating AI response...")
            ai_response = self.loop.run_until_complete(
                self.agent.get_ai_response(transcription, self.conversation_history)
            )
            print(f"🤖 [AI] Response: \"{ai_response}\"\n")

            # Convert to speech and send
            print("🔊 [PIPELINE] Converting response to speech...")
            response_audio = self.loop.run_until_complete(self.agent.text_to_speech(ai_response))
            if response_audio:
                print("📤 [PIPELINE] Sending audio response...")
                self.utils.send_audio_chunks(self.conn, response_audio)
                print("✅ [PIPELINE] Response sent! Listening for next input...\n")
            else:
                print("❌ [PIPELINE] Failed to generate response audio")

        except Exception as e:
            print(f"❌ [PIPELINE] Processing error: {e}")
            import traceback
            traceback.print_exc()

    def handle_client(self):
        """Handle AudioSocket client connection"""
        print(f"\n{'='*60}")
        print(f"🔗 [CONNECTION] New AudioSocket connection from {self.addr}")
        print(f"{'='*60}\n")

        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("⚡ [CONNECTION] TCP_NODELAY enabled for low latency")

        try:
            # Read UUID
            print("🆔 [SETUP] Waiting for call UUID...")
            msg_type, payload = self.utils.read_message(self.conn)
            if msg_type == MSG_UUID and len(payload) == 16:
                self.call_uuid = payload.hex()
                print(f"✅ [UUID] Call UUID received: {self.call_uuid}\n")
            else:
                print(f"❌ [SETUP] Expected UUID, got message type {msg_type}")
                return

            # Send greeting
            greeting = "Hello! I'm your AI assistant. How can I help you today?"
            print(f"👋 [GREETING] Sending: '{greeting}'")

            greeting_audio = self.loop.run_until_complete(self.agent.text_to_speech(greeting))
            if greeting_audio:
                print("📤 [GREETING] Audio generated successfully, sending to client...")
                self.utils.send_audio_chunks(self.conn, greeting_audio)
                print("✅ [GREETING] Greeting sent successfully\n")
            else:
                print("❌ [GREETING] Failed to generate greeting audio")
                return

            self.conversation_history.append({
                "role": "assistant",
                "content": greeting
            })

            print("🎧 [READY] Ready to receive audio from client...\n")

            # Main conversation loop
            audio_chunk_count = 0
            total_audio_received = 0

            while True:
                try:
                    msg_type, payload = self.utils.read_message(self.conn)
                    if msg_type is None:
                        print("🔌 [CONNECTION] Connection closed (read returned None)")
                        break

                    if msg_type == MSG_HANGUP:
                        print("📞 [HANGUP] Received hangup signal")
                        break

                    elif msg_type == MSG_AUDIO:
                        audio_chunk_count += 1
                        total_audio_received += len(payload)

                        if self.is_processing:
                            continue  # Skip audio while processing

                        # Add audio to VAD and check if speech is complete
                        should_process = self.vad.add_audio(payload)

                        if should_process:
                            self.is_processing = True

                            # Get the accumulated audio buffer
                            audio_data = self.vad.get_audio_buffer()

                            # Process in a separate thread to avoid blocking
                            processing_thread = threading.Thread(
                                target=self._process_audio_thread,
                                args=(audio_data,),
                                daemon=True
                            )
                            processing_thread.start()

                    elif msg_type == MSG_DTMF:
                        dtmf_digit = chr(payload[0]) if payload else ""
                        print(f"📞 [DTMF] Received DTMF digit: '{dtmf_digit}'")
                        if dtmf_digit == '*':
                            print("⭐ [DTMF] Star key pressed, ending call...")
                            farewell = "Goodbye! Have a great day!"
                            farewell_audio = self.loop.run_until_complete(self.agent.text_to_speech(farewell))
                            if farewell_audio:
                                self.utils.send_audio_chunks(self.conn, farewell_audio)
                            break

                    elif msg_type == MSG_ERROR:
                        error_code = payload[0] if payload else 0
                        print(f"🚨 [ERROR] AudioSocket error code: {error_code}")
                        break

                    else:
                        if DEBUG:
                            print(f"❓ [UNKNOWN] Unknown message type: {msg_type}")

                except Exception as e:
                    print(f"❌ [LOOP] Error in main loop: {e}")
                    continue

        except Exception as e:
            print(f"💥 [HANDLER] Client handler error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            print(f"\n{'='*60}")
            print(f"🔌 [DISCONNECT] Connection closed from {self.addr}")
            print(f"📊 [STATS] Total audio chunks received: {audio_chunk_count}")
            print(f"📊 [STATS] Total audio bytes received: {total_audio_received}")
            print(f"{'='*60}\n")

            self.loop.close()
            self.conn.close()

    def _process_audio_thread(self, audio_data):
        """Process audio in separate thread"""
        try:
            self.process_transcription(audio_data)
        finally:
            self.is_processing = False