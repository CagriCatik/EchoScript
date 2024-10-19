# EchoScript

EchoScript is a real-time speech-to-text transcription tool built with Python, using OpenAI's Whisper model for highly accurate transcription. The audio is captured from your microphone, processed, transcribed, and saved to a text file automatically.

## Features

- **Real-time audio capture:** Continuously records audio from the microphone in real-time.
- **Accurate transcription:** Uses the Whisper model for high-quality speech-to-text transcription.
- **Automatic file saving:** Saves the transcribed text to a `.txt` file after processing.

## Requirements

Ensure you have the following dependencies installed to run EchoScript:

- **Python 3.x**
- The dependencies listed in `requirements.txt`.

### Installing Dependencies

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## How to Run

1. The script will start recording audio from your microphone. To stop recording, press `Ctrl + C`.
2. After you stop the recording, the transcription process will begin automatically.
3. The transcription will be saved to a file named `recordings/transcription_output.txt`.

### Example Output

While running the script, you’ll see output similar to this in your terminal:

```bash
Recording... Press Ctrl + C to stop.
Transcription: This is an example transcription.
Transcription saved to transcription_output.txt
```

## Customization Options

- **Model selection:** Whisper offers different model sizes (e.g., `"small"`, `"medium"`, `"large"`). You can modify the model size to trade off between transcription speed and accuracy.

## Known Issues and Future Improvements

- **Microphone Check:** Ensure that your microphone is properly configured and functioning before running the script to avoid input issues.
- **Transcription Trigger:** The transcription process begins only after you stop the recording by pressing `Ctrl + C`. Be aware that no transcription will occur until the recording is stopped.
- **Performance Considerations:** Depending on your system's resources and the size of the Whisper model used, the transcription process may take longer to complete. Larger models provide better accuracy but require more processing power.
- **Project Growth Potential:** This project has great potential for expansion, and can serve as the foundation for a more complex, feature-rich application.
- **Future Feature Expansion:** There is significant room for further development. New features, such as support for multiple languages, real-time transcription display, or advanced audio processing, can be implemented to enhance its capabilities. 
