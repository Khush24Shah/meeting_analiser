# Meeting Analyzer

The Meeting Analyzer is a tool that allows you to analyze the sentiment transcription, minutes, and productivity of your meetings.

## Prerequisites

- [Ollama](https://ollama.com/download) must be installed and running.
- Run `ollama pull mistral` in the terminal to use the Mistral model.
- Run `pip install -r requirements.txt` to install the required packages.

## Usage

1. Run `streamlit run app.py` in the terminal.
2. Upload the meeting audio file.
3. The tool will ask for segment duration in seconds.
4. The tool will generate the transcription for each segment.
5. The point-wise summary of the meeting will be displayed.
6. Overall and segment-wise sentiment analysis will be displayed.
7. According to the user-provided keywords, the tool will display the productivity of the meeting.