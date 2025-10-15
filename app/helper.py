import numpy as np

class Helper:

    def __init__(self):
        pass

    def calculate_rms(self, audio_data):
        """Calculate RMS (Root Mean Square) of audio data"""
        if len(audio_data) == 0:
            return 0.0
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) == 0:
                return 0.0
            rms = float(np.sqrt(np.mean(audio_array.astype(np.float64)**2)))
            return rms if not np.isnan(rms) else 0.0
        except Exception as e:
            return 0.0

    def normalize_audio(self, audio_data):
        """Normalize audio to increase volume"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            if len(audio_array) == 0:
                return audio_data
            
            # Calculate current peak
            peak = np.max(np.abs(audio_array))
            if peak > 0:
                # Normalize to 80% of max to avoid clipping
                target_peak = 32768 * 0.8
                gain = target_peak / peak
                # Limit gain to avoid amplifying noise too much
                gain = min(gain, 10.0)
                audio_array = audio_array * gain
                audio_array = np.clip(audio_array, -32768, 32767)
            
            return audio_array.astype(np.int16).tobytes()
        except Exception as e:
            print(f"[!] Normalization error: {e}")
            return audio_data
        