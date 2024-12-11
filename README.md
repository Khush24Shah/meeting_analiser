# Meeting Analyzer

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
6. Overall and segment-wise sentiment analysis will be displayed.
7. According to the user-provided keywords, the tool will display the productivity of the meeting.

## Demo

