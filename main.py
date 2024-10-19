import pyaudio
import numpy as np
import whisper
import os

# Audio settings
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate in Hz
CHUNK = 1024  # Size of audio chunks

# Function to capture real-time audio
def capture_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording... Press Ctrl+C to stop.")

    frames = []
    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(np.frombuffer(data, dtype=np.int16))
    except KeyboardInterrupt:
        print("Stopped recording.")
        stream.stop_stream()
        stream.close()
        p.terminate()

    return np.concatenate(frames)

# Function to process audio data for Whisper
def process_audio_data(raw_audio):
    float_audio = raw_audio / np.iinfo(np.int16).max
    return float_audio.astype(np.float32)

# Function to perform transcription
def transcribe_audio(model, audio):
    result = model.transcribe(audio)
    return result["text"]

# Function for saving transcription to a .txt file
def save_transcription_to_file(text, file_path="./recordings/transcription_output.txt"):
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Transcription saved to {file_path}")

# Function for real-time transcription and saving to file
def real_time_transcription():
    model = whisper.load_model("small")
    audio_data = capture_audio()
    processed_audio = process_audio_data(audio_data)

    try:
        transcription = transcribe_audio(model, processed_audio)
        print("Transcription:", transcription)
        save_transcription_to_file(transcription, "./recordings/transcription_output.txt")
    except KeyboardInterrupt:
        print("Transcription interrupted.")

# Run the real-time transcription and save the result
if __name__ == "__main__":
    try:
        real_time_transcription()
    except KeyboardInterrupt:
        print("\nExiting transcription.")
