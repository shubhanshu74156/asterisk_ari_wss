import audioop

def handle_ai_call(connection):
    """Process bidirectional audio with AI agent"""
    print(f"Handling call with UUID: {connection.uuid}")
    
    while connection.connected:
        # Receive audio from Asterisk (16-bit, 8kHz, mono PCM)
        audio_in = connection.read()
        
        if audio_in is None:
            break
            
        # Send to AI agent for processing
        # Your AI agent should process this audio
        ai_response_audio = process_with_ai_agent(audio_in)
        
        # Send AI response back to Asterisk
        if ai_response_audio:
            # Ensure audio is 16-bit, 8kHz, mono PCM
            connection.write(ai_response_audio)
    
    print("Call ended")

def process_with_ai_agent(audio_data):
    """
    Send audio to your AI agent and get response
    Audio format: signed linear 16-bit, 8kHz, mono PCM (little-endian)
    
    If your AI needs different format, use audioop to convert:
    - Resample: audioop.ratecv()
    - Convert stereo/mono: audioop.tomono() / audioop.tostereo()
    - Convert to/from ulaw: audioop.ulaw2lin() / audioop.lin2ulaw()
    """
    # Example: Convert to 16kHz if your AI needs it
    # audio_16k, _ = audioop.ratecv(audio_data, 2, 1, 8000, 16000, None)
    
    # Send to your AI model (e.g., via WebSocket, HTTP, etc.)
    # ai_response = your_ai_client.process(audio_data)
    
    # Return response audio (must be 16-bit, 8kHz, mono PCM)
    return audio_data  # Replace with actual AI response
