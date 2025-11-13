
import socket
import threading
import wave
import struct
import numpy as np
import asyncio
import os
import tempfile
import time
from openai import OpenAI
from dotenv import load_dotenv
import random
import json
import statistics


# Load environment
load_dotenv()
print("🌍 Environment Variables Loaded")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("🤖 OpenAI Client Initialized")

DEBUG = True

HOST = "0.0.0.0"
PORT = 9092

# AudioSocket message types
MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10
MSG_ERROR = 0xff


class FunctionLibrary:
    """Library of available functions that AI can call"""

    def __init__(self):
        print("🔧 Function Library initialized")

    def end_call(self, reason="User requested"):
        """End the current call

        Args:
            reason (str): Reason for ending the call
        """
        print(f"📞 [FUNCTION] End call function initiated - Reason: {reason}")
        result = {"action": "end_call", "reason": reason, "message": f"Call ended: {reason}"}
        print("✅ [FUNCTION] End call function executed")
        return result

    def roll_dice(self, num_dice=1, sides=6):
        """Roll one or more dice

        Args:
            num_dice (int): Number of dice to roll (default: 1)
            sides (int): Number of sides on each die (default: 6)
        """
        print(f"🎲 [FUNCTION] Rolling {num_dice} dice with {sides} sides each")

        results = []
        for i in range(num_dice):
            roll = random.randint(1, sides)
            results.append(roll)

        total = sum(results)
        result = {
            "action": "roll_dice",
            "num_dice": num_dice,
            "sides": sides,
            "results": results,
            "total": total,
            "message": f"Rolled {results} (total: {total})" if num_dice > 1 else f"Rolled {results[0]}"
        }

        print(f"🎯 [FUNCTION] Dice results: {result['message']}")
        print("✅ [FUNCTION] Roll dice executed")
        return result

    def pick_random_animal(self, category="all"):
        """Pick a random animal from a specified category

        Args:
            category (str): Category of animal (all, mammals, birds, reptiles, etc.)
        """
        print(f"🐾 [FUNCTION] Picking random animal from category: {category}")

        animals = {
            "all": ["Lion", "Elephant", "Cheetah", "Dog", "Cat", "Pig", "Sparrow", "Eagle", "Snake", "Turtle", "Shark", "Dolphin"],
            "mammals": ["Lion", "Elephant", "Cheetah", "Dog", "Cat", "Pig", "Dolphin", "Bear", "Tiger"],
            "birds": ["Sparrow", "Eagle", "Parrot", "Penguin", "Owl", "Flamingo"],
            "reptiles": ["Snake", "Turtle", "Lizard", "Crocodile", "Iguana"],
            "sea": ["Shark", "Dolphin", "Whale", "Octopus", "Starfish"]
        }

        animal_list = animals.get(category.lower(), animals["all"])
        selected_animal = random.choice(animal_list)

        result = {
            "action": "pick_random_animal",
            "category": category,
            "selected_animal": selected_animal,
            "message": f"Selected {selected_animal} from {category} animals"
        }

        print(f"🦁 [FUNCTION] Selected animal: {selected_animal}")
        print("✅ [FUNCTION] Animal selection executed")
        return result

    def get_random_number(self, min_value=1, max_value=100):
        """Generate a random number within a specified range

        Args:
            min_value (int): Minimum value (default: 1)
            max_value (int): Maximum value (default: 100)
        """
        print(f"🔢 [FUNCTION] Generating random number between {min_value} and {max_value}")

        number = random.randint(min_value, max_value)

        result = {
            "action": "get_random_number",
            "min_value": min_value,
            "max_value": max_value,
            "number": number,
            "message": f"Generated random number: {number}"
        }

        print(f"🎯 [FUNCTION] Random number: {number}")
        print("✅ [FUNCTION] Random number generation executed")
        return result

    def flip_coin(self, num_flips=1):
        """Flip one or more coins

        Args:
            num_flips (int): Number of coins to flip (default: 1)
        """
        print(f"🪙 [FUNCTION] Flipping {num_flips} coin(s)")

        results = []
        for i in range(num_flips):
            flip = random.choice(["Heads", "Tails"])
            results.append(flip)

        result = {
            "action": "flip_coin",
            "num_flips": num_flips,
            "results": results,
            "message": f"Coin flip results: {', '.join(results)}" if num_flips > 1 else f"Coin flip: {results[0]}"
        }

        print(f"🎯 [FUNCTION] Coin flip results: {results}")
        print("✅ [FUNCTION] Coin flip executed")
        return result

    def get_available_functions(self):
        """Get list of all available functions"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "end_call",
                    "description": "End the current phone call. Use when user says goodbye, wants to hang up, or indicates they're done.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Reason for ending the call"
                            }
                        },
                        "required": ["reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "roll_dice",
                    "description": "Roll one or more dice. Use when user wants to roll dice, play dice games, or needs random numbers via dice.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "num_dice": {
                                "type": "integer",
                                "description": "Number of dice to roll",
                                "default": 1
                            },
                            "sides": {
                                "type": "integer",
                                "description": "Number of sides on each die",
                                "default": 6
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "pick_random_animal",
                    "description": "Pick a random animal from a category. Use when user wants to know about animals, play animal games, or needs a random animal name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Category of animal to pick from",
                                "enum": ["all", "mammals", "birds", "reptiles", "sea"],
                                "default": "all"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_random_number",
                    "description": "Generate a random number within a range. Use when user needs random numbers, lottery numbers, or number games.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "min_value": {
                                "type": "integer",
                                "description": "Minimum value for the random number",
                                "default": 1
                            },
                            "max_value": {
                                "type": "integer",
                                "description": "Maximum value for the random number",
                                "default": 100
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "flip_coin",
                    "description": "Flip one or more coins to get heads or tails. Use when user wants to flip coins, make decisions, or play coin games.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "num_flips": {
                                "type": "integer",
                                "description": "Number of coins to flip",
                                "default": 1
                            }
                        }
                    }
                }
            }
        ]

    def execute_function(self, function_name, arguments):
        """Execute a function by name with given arguments"""
        try:
            function_map = {
                "end_call": self.end_call,
                "roll_dice": self.roll_dice,
                "pick_random_animal": self.pick_random_animal,
                "get_random_number": self.get_random_number,
                "flip_coin": self.flip_coin
            }

            if function_name in function_map:
                return function_map[function_name](**arguments)
            else:
                return {"error": f"Function {function_name} not found"}

        except Exception as e:
            print(f"❌ [FUNCTION] Error executing {function_name}: {e}")
            return {"error": f"Error executing {function_name}: {str(e)}"}


class SimpleVAD:
    """Simple Voice Activity Detection using RMS and silence detection"""

    def __init__(self):
        self.audio_buffer = bytearray()
        self.silence_threshold = 200  # RMS threshold for silence
        self.speech_threshold = 500   # RMS threshold for speech
        self.min_speech_duration = 0.2  # seconds
        self.silence_duration = 1.5     # seconds of silence to trigger processing
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = time.time()
        self.processing_interrupted = False
        self.interrupted_buffer = bytearray()
        self.rms = []
        print("🎤 Simple VAD initialized")

    def calculate_rms(self, audio_data):
        """Calculate RMS of audio data"""
        if len(audio_data) == 0:
            return 0.0
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) == 0:
                return 0.0
            rms = float(np.sqrt(np.mean(audio_array.astype(np.float64)**2)))
            return rms if not np.isnan(rms) else 0.0
        except:
            return 0.0

    def add_audio(self, audio_chunk, is_processing=False):
        """Add audio chunk and return speech status"""
        self.audio_buffer.extend(audio_chunk)

        # Calculate RMS of the chunk
        rms = self.calculate_rms(audio_chunk)
        self.rms.append(rms)

        result = self.detect_spike_point(self.rms)
        if result:
            self.speech_threshold = result[1]


        current_time = time.time()

        # Determine if currently speaking
        if rms >= self.speech_threshold:
            if not self.is_speaking:
                if is_processing:
                    print("🔄 [VAD] 🔴 Speech detected during processing - Interrupting and accumulating")
                    self.processing_interrupted = True
                    self.interrupted_buffer.extend(self.audio_buffer)
                else:
                    print("🎙️ [VAD] 🔴 Speech detected - Recording started")
                self.is_speaking = True
                self.speech_start_time = current_time
            self.last_speech_time = current_time

        elif self.is_speaking and rms < self.silence_threshold:

            # Check if we've been silent long enough
            silence_duration = current_time - self.last_speech_time
            if silence_duration >= self.silence_duration:
                speech_duration = self.last_speech_time - self.speech_start_time

                if speech_duration >= self.min_speech_duration:
                    if self.processing_interrupted:
                        print(f"🎙️ [VAD] ⏹️ Interrupted speech ended after {speech_duration:.1f}s - Processing combined audio...")
                        self.is_speaking = False  # ✅ CORRECT: Mark as not speaking
                        return True               # ✅ CORRECT: Trigger processing of combined audio
                    else:
                        print(f"🎙️ [VAD] ⏹️ Speech ended after {speech_duration:.1f}s - Processing...")
                        self.is_speaking = False
                        return True  # Signal to process speech
                else:
                    print(f"🎙️ [VAD] ⚠️ Speech too short ({speech_duration:.1f}s) - Ignoring")
                    self.is_speaking = False
                    if not is_processing:
                        self.audio_buffer.clear()

        # Log RMS periodically
        if DEBUG and len(self.audio_buffer) % 16000 == 0:
            duration = len(self.audio_buffer) / (8000 * 2)
            status = "Speaking" if self.is_speaking else "Silent"
            if is_processing and self.is_speaking:
                status = "Speaking+Processing(Interrupted)"
            elif is_processing:
                status = "Processing"
            print(f"📊 [VAD] Buffer: {duration:.1f}s, RMS: {rms:.0f}, Status: {status}")

        return False
    

    def detect_spike_point(self, data, m=3, T_S=3, T_R=10, eps=1):
        """
        Detect the first index where numbers 'start spinning' (large deviation).

        Parameters:
            data : list of numbers
            m    : warm-up window (minimum points before detection)
            T_S  : threshold for robust z-score (default 3)
            T_R  : threshold for relative ratio (default 10)
            eps  : small value to prevent division by zero

        Returns:
            (index, value) where spinning starts, or None if not found.
        """
        n = len(data)
        if n <= m:
            return None  # not enough data
        
        for k in range(m, n):
            previous = data[:k]
            B = statistics.median(previous)
            MAD = statistics.median([abs(x - B) for x in previous]) or eps
            
            S = abs(data[k] - B) / max(MAD, eps)   
            R = data[k] / max(B, eps)              
            
            if S > T_S or R > T_R:
                return k, data[k]
        
        return None

    def get_audio_buffer(self):
        """Get and clear the audio buffer, combining with interrupted audio if any"""
        if self.processing_interrupted and len(self.interrupted_buffer) > 0:
            combined_audio = bytes(self.interrupted_buffer) + bytes(self.audio_buffer)
            print(f"🔗 [VAD] Combining interrupted audio: {len(self.interrupted_buffer)} + {len(self.audio_buffer)} = {len(combined_audio)} bytes")
            self.interrupted_buffer.clear()
        else:
            combined_audio = bytes(self.audio_buffer)

        self.audio_buffer.clear()
        self.processing_interrupted = False
        return combined_audio


class Utils:
    def __init__(self):
        print("🔧 Utils Class Initialized")

    def send_audio_chunks(self, conn, audio_data, handler_ref=None):
        """Send audio data in 320-byte chunks via AudioSocket protocol with interruption support

        Args:
            conn: Socket connection
            audio_data: Audio data to send
            handler_ref: Reference to handler for checking interruption status
        """
        chunk_size = 320
        total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size

        print(f"📤 [SEND] Sending {len(audio_data)} bytes in {total_chunks} chunks")

        chunks_sent = 0
        for i in range(0, len(audio_data), chunk_size):
            # Check for interruption before sending each chunk
            if handler_ref and handler_ref.audio_sending_cancelled:
                print(f"🚫 [SEND] Audio sending cancelled due to interruption at chunk {chunks_sent + 1}/{total_chunks}")
                print(f"⏹️ [SEND] Skipped remaining {total_chunks - chunks_sent} chunks")
                return False  # Return False to indicate cancellation

            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk += b'\x00' * (chunk_size - len(chunk))
            header = struct.pack('B', MSG_AUDIO) + struct.pack('>H', len(chunk))

            try:
                conn.sendall(header + chunk)
                time.sleep(0.02)
                chunks_sent += 1

                if DEBUG and i % (chunk_size * 10) == 0:  # Every 10th chunk
                    print(f"📡 Sent chunk {chunks_sent}/{total_chunks}")

            except Exception as e:
                print(f"❌ Error sending audio chunk {chunks_sent + 1}: {e}")
                return False

        print(f"✅ [SEND] Audio transmission complete ({total_chunks} chunks)")
        return True  # Return True to indicate successful completion

    def read_message(self, conn, timeout=10):
        """Read one full AudioSocket message"""
        conn.settimeout(timeout)
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
                if msg_type != MSG_AUDIO:
                    print(f"📥 [READ] {msg_name}: {payload_length} bytes")

            payload = b''
            while len(payload) < payload_length:
                chunk = conn.recv(payload_length - len(payload))
                if not chunk:
                    break
                payload += chunk

            return msg_type, payload
        except socket.timeout:
            print("Socket timeout")
            return None, None
        except Exception as e:
            print(f"❌ Error reading message: {e}")
            return None, None


class Agent:
    def __init__(self, custom_function_prompt=""):
        self.function_library = FunctionLibrary()
        self.custom_function_prompt = custom_function_prompt
        print("🤖 Agent Class Initialized with AI-driven function calling")

    async def transcribe_audio(self, audio_bytes, sample_rate=8000):
        """Async transcription using OpenAI Whisper"""
        temp_path = None
        try:
            print(f"🎙️ [TRANSCRIBE] Starting transcription of {len(audio_bytes)} bytes...")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_path = temp_audio.name

            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)

            print(f"💾 [TRANSCRIBE] WAV file created: {os.path.basename(temp_path)}")

            loop = asyncio.get_event_loop()
            with open(temp_path, 'rb') as audio_file:
                print("📡 [TRANSCRIBE] Sending to OpenAI Whisper API...")
                transcript = await loop.run_in_executor(
                    None, 
                    lambda: client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en"
                    )
                )

            result = transcript.text.strip()
            print(f"✅ [TRANSCRIBE] Success: '{result}'")
            return result

        except Exception as e:
            print(f"❌ [TRANSCRIBE] Error: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    print(f"🗑️ [TRANSCRIBE] Cleaned up temp file")
                except:
                    pass

    async def get_ai_response_with_functions(self, user_text, conversation_history):
        """Get AI response with intelligent function calling"""
        try:
            print(f"🧠 [CHAT] Processing user input with function calling: '{user_text}'")

            # Create system message with function calling instructions
            system_message = {
                "role": "system",
                "content": f"""You are a helpful AI phone assistant. Keep your responses brief, clear, and conversational. 
                Use simple language suitable for phone conversations. Keep responses under 3 sentences.

                You have access to several functions that you can call when appropriate:
                - Roll dice when users want to play games, make random choices, or need dice rolls
                - Pick random animals when users are curious about animals or playing animal games  
                - Generate random numbers for lottery numbers, games, or random choices
                - Flip coins for decision making or coin toss games
                - End calls when users say goodbye or want to hang up

                Additional function calling guidance:
                {self.custom_function_prompt}

                Always respond naturally and incorporate function results into your conversation."""
            }

            # Add the system message if not already present or update it
            if not conversation_history or conversation_history[0]["role"] != "system":
                conversation_history.insert(0, system_message)
            else:
                conversation_history[0] = system_message

            conversation_history.append({
                "role": "user",
                "content": user_text
            })

            print("📡 [CHAT] Sending to OpenAI Chat API with function calling...")

            # Get available functions
            available_functions = self.function_library.get_available_functions()

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=conversation_history,
                    temperature=0.7,
                    max_tokens=150,
                    tools=available_functions,
                    tool_choice="auto"
                )
            )

            message = response.choices[0].message

            # Check if AI wants to call functions
            if message.tool_calls:
                print(f"🔧 [FUNCTIONS] AI decided to call {len(message.tool_calls)} function(s)")

                # Add assistant message with tool calls to conversation
                conversation_history.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [tc.dict() for tc in message.tool_calls]
                })

                # Execute each function call
                function_results = []
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"🎯 [FUNCTION] Executing {function_name} with args: {function_args}")

                    # Execute the function
                    result = self.function_library.execute_function(function_name, function_args)
                    function_results.append(result)

                    # Add function result to conversation
                    conversation_history.append({
                        "role": "tool",
                        "content": json.dumps(result),
                        "tool_call_id": tool_call.id
                    })

                # Get final response incorporating function results
                print("🔄 [CHAT] Getting final response with function results...")
                final_response = await loop.run_in_executor(
                    None,
                    lambda: client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=conversation_history,
                        temperature=0.7,
                        max_tokens=150
                    )
                )

                ai_text = final_response.choices[0].message.content
                print(f"✅ [CHAT] Final AI Response with functions: '{ai_text}'")

                # Check if any function was "end_call"
                end_call_requested = any(result.get("action") == "end_call" for result in function_results)

                return ai_text, function_results, end_call_requested

            else:
                # No function calls, just regular response
                ai_text = message.content
                print(f"✅ [CHAT] Regular AI Response: '{ai_text}'")

                conversation_history.append({
                    "role": "assistant",
                    "content": ai_text
                })

                return ai_text, [], False

        except Exception as e:
            print(f"❌ [CHAT] Error: {e}")
            import traceback
            traceback.print_exc()
            return "I'm sorry, I had trouble processing that. Could you please repeat?", [], False

    async def text_to_speech(self, text):
        """Convert text to speech using OpenAI TTS (async)"""
        try:
            print(f"🔊 [TTS] Converting to speech: '{text}'")
            print("📡 [TTS] Sending to OpenAI TTS API...")

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=text,
                    response_format="pcm",
                    speed=1.0
                )
            )

            print("✅ [TTS] Received audio response from OpenAI")

            # OpenAI TTS outputs 24kHz PCM, downsample to 8kHz
            audio_data = response.content
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Downsample from 24kHz to 8kHz
            import librosa
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            downsampled = librosa.resample(audio_array, orig_sr=24000, target_sr=8000)
            result = (downsampled * 32768).astype(np.int16).tobytes()
            print(f"🔄 [TTS] Downsampled audio: {len(downsampled)} samples at 8kHz ({len(result)} bytes)")

            return result
        except Exception as e:
            print(f"❌ [TTS] Error: {e}")
            return None


class AdvancedHandler:
    def __init__(self, conn, addr, custom_function_prompt=""):
        self.conn = conn
        self.addr = addr
        self.utils = Utils()
        self.agent = Agent(custom_function_prompt)
        self.vad = SimpleVAD()
        self.conversation_history = []
        self.is_processing = False
        self.call_uuid = None
        self.processing_cancelled = False
        self.audio_sending_cancelled = False  
        self.is_sending_audio = False 

        # Create event loop for async operations
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def cancel_audio_sending(self):
        """Cancel any ongoing audio transmission"""
        if self.is_sending_audio:
            print("🚫 [AUDIO] Cancelling ongoing audio transmission")
            self.audio_sending_cancelled = True
        else:
            print("ℹ️ [AUDIO] No audio transmission to cancel")

    def process_transcription(self, audio_data):
        """Process the audio data and generate AI response with function calling"""
        try:
            if self.processing_cancelled:
                print("🚫 [PIPELINE] Processing cancelled due to interruption")
                return

            print(f"🔄 [PIPELINE] Processing {len(audio_data)} bytes of audio...")

            # Transcribe audio
            transcription = self.loop.run_until_complete(
                self.agent.transcribe_audio(audio_data)
            )

            if self.processing_cancelled:
                print("🚫 [PIPELINE] Processing cancelled during transcription")
                return

            if not transcription or len(transcription.strip()) == 0:
                print("⚠️ [PIPELINE] Empty transcription received")
                return

            print(f"👤 [USER] Transcription: \"{transcription}\"\n")

            # Check for manual goodbye phrases (fallback)
            goodbye_phrases = ['bye', 'goodbye', 'bye-bye', 'see you', 'farewell']
            if any(phrase in transcription.lower() for phrase in goodbye_phrases):
                print("👋 [GOODBYE] Manual goodbye detected, ending conversation...")
                farewell = "Goodbye! Have a great day!"
                print(f"🤖 [AI] Final response: '{farewell}'")
                farewell_audio = self.loop.run_until_complete(self.agent.text_to_speech(farewell))
                if farewell_audio:
                    self.utils.send_audio_chunks(self.conn, farewell_audio, self)
                self.conn.close()
                return

            if self.processing_cancelled:
                print("🚫 [PIPELINE] Processing cancelled before AI response")
                return

            # Get AI response with intelligent function calling
            print("🧠 [PIPELINE] Getting AI response with function calling...")
            ai_response, function_results, end_call_requested = self.loop.run_until_complete(
                self.agent.get_ai_response_with_functions(transcription, self.conversation_history)
            )

            if self.processing_cancelled:
                print("🚫 [PIPELINE] Processing cancelled after AI response")
                return

            print(f"🤖 [AI] Response: \"{ai_response}\"")
            if function_results:
                print(f"🔧 [FUNCTIONS] Executed {len(function_results)} function(s)")
                for result in function_results:
                    print(f"   - {result.get('action', 'unknown')}: {result.get('message', 'no message')}")

            # Handle end call request from AI
            if end_call_requested:
                print("📞 [AI-ENDING] AI decided to end the call")
                farewell_audio = self.loop.run_until_complete(self.agent.text_to_speech(ai_response))
                if farewell_audio:
                    self.utils.send_audio_chunks(self.conn, farewell_audio, self)
                self.conn.close()
                return

            # Convert to speech and send
            print("🔊 [PIPELINE] Converting response to speech...")
            response_audio = self.loop.run_until_complete(self.agent.text_to_speech(ai_response))

            if not self.processing_cancelled and response_audio:
                print("📤 [PIPELINE] Sending audio response...")

                # Reset audio cancellation flag and set sending status
                self.audio_sending_cancelled = False
                self.is_sending_audio = True

                # Send audio with interruption support
                audio_sent_successfully = self.utils.send_audio_chunks(self.conn, response_audio, self)

                # Reset sending status
                self.is_sending_audio = False

                if audio_sent_successfully:
                    print("✅ [PIPELINE] Response sent! Listening for next input...\n")
                else:
                    print("🚫 [PIPELINE] Audio transmission was interrupted")

            elif self.processing_cancelled:
                print("🚫 [PIPELINE] Processing cancelled - not sending audio")
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

                print("⏳ [SETUP] Waiting 2 seconds for Asterisk to establish call...")
                time.sleep(5)
                print("✅ [SETUP] Call established, ready to send greeting\n")

            else:
                print(f"❌ [SETUP] Expected UUID, got message type {msg_type}")
                return

          
            greeting = "Hello! I'm your AI assistant. What can I do for you?"
            print(f"👋 [GREETING] Sending: '{greeting}'")

            greeting_audio = self.loop.run_until_complete(self.agent.text_to_speech(greeting))
            if greeting_audio:
                print("📤 [GREETING] Audio generated successfully, sending to client...")
                self.is_sending_audio = True
                audio_sent = self.utils.send_audio_chunks(self.conn, greeting_audio, self)
                self.is_sending_audio = False

                if audio_sent:
                    print("✅ [GREETING] Greeting sent successfully\n")
                else:
                    print("🚫 [GREETING] Greeting transmission interrupted\n")
            else:
                print("❌ [GREETING] Failed to generate greeting audio")
                return

            self.greeting_sent = True

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

                        # NEW: Handle speech interruption during processing OR audio sending
                        if (self.is_processing and self.vad.is_speaking) or (self.is_sending_audio and self.vad.is_speaking):
                            if self.is_processing:
                                print("🔄 [INTERRUPT] User started speaking during processing - Cancelling current processing")
                                self.processing_cancelled = True

                            if self.is_sending_audio:
                                print("🔇 [INTERRUPT] User started speaking during audio playback - Cancelling audio transmission")
                                self.cancel_audio_sending()

                            continue

                        # Add audio to VAD and check if speech is complete
                        should_process = self.vad.add_audio(payload, is_processing=self.is_processing)

                        if should_process:
                            if self.is_processing:
                                print("🚫 [INTERRUPT] Cancelling previous processing for new/combined audio")
                                self.processing_cancelled = True
                                time.sleep(0.1)

                            if self.is_sending_audio:
                                print("🚫 [INTERRUPT] Cancelling audio transmission for new processing")
                                self.cancel_audio_sending()
                                time.sleep(0.1)

                            self.is_processing = True
                            self.processing_cancelled = False
                            self.audio_sending_cancelled = False  # Reset audio cancellation

                            # Get the accumulated audio buffer
                            audio_data = self.vad.get_audio_buffer()
                            # self.vad.clear_buffer()

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

                            # Cancel any ongoing audio transmission
                            if self.is_sending_audio:
                                self.cancel_audio_sending()

                            farewell = "Goodbye! Have a great day!"
                            farewell_audio = self.loop.run_until_complete(self.agent.text_to_speech(farewell))
                            if farewell_audio:
                                self.is_sending_audio = True
                                self.utils.send_audio_chunks(self.conn, farewell_audio, self)
                                self.is_sending_audio = False
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
                    import traceback
                    traceback.print_exc()
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
            self.processing_cancelled = False


def main():
    print(f"\n🚀 Starting Enhanced AI Function Calling AudioSocket Server")
    print(f"🌐 Host: {HOST}")
    print(f"🔌 Port: {PORT}")
    print(f"🧠 Feature: AI-driven intelligent function calling")
    print(f"🔇 Feature: Voice interruption with AI audio cancellation")
    print(f"🎲 Available Functions: Roll Dice, Random Animal, Random Number, Flip Coin, End Call")
    print(f"{'='*60}")

    # You can customize the function calling behavior here
    custom_prompt = """
    Be proactive about using functions when they would be helpful or fun:
    - If someone mentions games, luck, or randomness, offer to roll dice or flip coins
    - If they talk about animals or nature, suggest picking a random animal
    - If they need numbers for anything, use the random number generator
    - Be natural and conversational about function usage
    - Keep responses concise since users can interrupt
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print("🔧 Socket created")
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print("⚙️ Socket options configured (SO_REUSEADDR)")
        s.bind((HOST, PORT))
        print(f"🔗 Socket bound to {HOST}:{PORT}")
        s.listen()
        print("👂 Socket listening for connections")
        print(f"\n✅ Server ready! Waiting for Asterisk connections...\n")

        connection_count = 0

        while True:
            try:
                conn, addr = s.accept()
                connection_count += 1
                print(f"\n🎉 [SERVER] New connection #{connection_count} from {addr}")

                # Create handler instance for this connection with custom function prompt
                handler = AdvancedHandler(conn, addr, custom_prompt)

                # Create new handler thread
                handler_thread = threading.Thread(
                    target=handler.handle_client, 
                    daemon=True,
                    name=f"Handler-{connection_count}"
                )
                handler_thread.start()
                print(f"🧵 [SERVER] Started handler thread: {handler_thread.name}")

            except KeyboardInterrupt:
                print("\n⛔ [SERVER] Keyboard interrupt received")
                print("🛑 [SERVER] Shutting down gracefully...")
                break
            except Exception as e:
                print(f"💥 [SERVER] Server error: {e}")
                import traceback
                traceback.print_exc()  


if __name__ == "__main__":
    main()
