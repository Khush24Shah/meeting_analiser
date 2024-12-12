# Meeting Analyzer ([Repository](https://github.com/Khush24Shah/meeting_analyser))

Meeting Analyzer is a tool that allows you to analyze the sentiment transcription, minutes, and productivity of your meetings.

## Prerequisites

- [Ollama](https://ollama.com/download) must be installed and running. If it is closed, the minutes of the meeting will fail to generate and hence the application will not work.
- Run `ollama pull mistral` in the terminal to use the Mistral model.
- Run `pip install -r requirements.txt` to install the required packages.
- The device running the tool must be connected to the internet.

## Usage

1. Run `streamlit run app.py` in the terminal. We use [Streamlit](https://streamlit.io/) for UI.
2. Upload the meeting audio file. (Two sample audio files are provided in [sample_audios](https://github.com/Khush24Shah/meeting_analyser/tree/main/sample_audios).)
3. The tool will ask for segment duration in seconds.
4. The tool will generate the transcription for each segment.
5. The point-wise summary of the meeting will be displayed.
6. Overall and segment-wise sentiment (Positive, Negative or Neutral) analysis will be displayed.
7. According to the user-provided keywords, the tool will display the productivity of the meeting.

## Demo

https://github.com/user-attachments/assets/10501faa-37d2-4cee-a4dc-1ece1baae7a9

## Approach

- In the function `preprocess_audio`, we preprocess the audio file by normalising the audio and filtering out the noise.
- In the function `split_audio`, we split the audio into segments of the user-provided duration. We do this to generate the transcription for each segment (smaller tasks are easier to handle) and to analyze the sentiment of each segment.
- In the function `evaluate_productivity`, we evaluate the productivity of the meeting based on the user-provided keywords. We focus on the sentences that contain these keywords.
- In the function `sentiment_analysis`, we analyze the sentiment of the meeting. We use `SentimentIntensityAnalyzer` from the `nltk` library to analyze the sentiment of each segment.
- To transcribe the audio, we use the `Whisper` model from OpenAI.
- For preparing the minutes of the meeting, we utilise the `Mistral` model provided by Ollama.
- Lastly, we use Streamlit to create the UI.