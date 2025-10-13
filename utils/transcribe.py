# import whisper

# model = whisper.load_model("turbo")

# # load audio and pad/trim it to fit 30 seconds
# audio = whisper.load_audio("recorded_6cfc4913c60f4b209e7362bb839e03d5.wav")
# audio = whisper.pad_or_trim(audio)

# # make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# # decode the audio
# options = whisper.DecodingOptions()
# result = whisper.decode(model, mel, options)

# # print the recognized text
# print(result.text)

# main.py (python example)

import os

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

# Path to the audio file
AUDIO_FILE = "working/recorded_f83b292d9008429aa56939b459d9622e.wav"

def main():
    try:
        # STEP 1 Create a Deepgram client using the API key
        deepgram = DeepgramClient()

        with open(AUDIO_FILE, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        #STEP 2: Configure Deepgram options for audio analysis
        options = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
        )

        # STEP 3: Call the transcribe_file method with the text payload and options
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

        # STEP 4: Print the response
        print(response.to_json(indent=4))

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()
