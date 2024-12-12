# Link to the repository: https://github.com/Khush24Shah/meeting_analyser

import streamlit as st
import librosa
import soundfile as sf
import os
import numpy as np
from scipy.signal import medfilt
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langchain_community.llms import Ollama
from ollama import chat
from ollama import ChatResponse

# Download NLTK data
nltk.download('punkt')

# Function to preprocess audio
def preprocess_audio(file_name, output_dir="preprocessed"):
    """
    Function to preprocess audio file

    Parameters:
    file_name (str): Path to the audio file
    output_dir (str): Path to the output directory

    Returns:
    preprocessed_file (str): Path to the preprocessed audio file
    """
    audio, sr = librosa.load(file_name, mono=True) # Load the audio file
    audio = audio / np.max(np.abs(audio))  # Normalize the audio
    audio_denoised = medfilt(audio, kernel_size=3)  # Basic noise reduction
    if not os.path.exists(output_dir): # Create the output directory if it doesn't exist
        os.makedirs(output_dir)
    preprocessed_file = os.path.join(output_dir, f"preprocessed_{os.path.basename(file_name)}") # Save the preprocessed audio
    sf.write(preprocessed_file, audio_denoised, sr) # Save the preprocessed audio
    return preprocessed_file

# Function to split audio into segments
def split_audio(file_name, segment_duration, output_dir="out"):
    """
    Function to split audio file into segments

    Parameters:
    file_name (str): Path to the audio file
    segment_duration (int): Duration of each segment in seconds
    output_dir (str): Path to the output directory

    Returns:
    split_dir (str): Path to the directory containing the audio segments
    """
    audio, sr = librosa.load(file_name) # Load the audio file
    # audio = librosa.effects.time_stretch(audio, rate=0.95)  # Slow down the audio
    buffer = segment_duration * sr # Buffer size for each segment
    samples_total = len(audio) # Total number of samples in the audio
    samples_written = 0 # Number of samples written
    counter = 1 # Counter for the segments
    file_base_name = os.path.splitext(os.path.basename(file_name))[0] # Base name of the file
    split_dir = os.path.join(output_dir, file_base_name) # Output directory for the segments
    
    if not os.path.exists(split_dir): # Create the output directory if it doesn't exist
        os.makedirs(split_dir)
    else: # Clear the output directory if it already exists so that we don't have any old files
        for file in os.listdir(split_dir):
            os.remove(os.path.join(split_dir, file))

    while samples_written < samples_total: # Loop until all samples are written
        buffer = min(samples_total - samples_written, buffer) # Update the buffer size when we reach the end of the audio

        block = audio[samples_written: (samples_written + buffer)] # Extract the block of samples for the segment
        out_filename = os.path.join(split_dir, f"split_{counter}.wav") # Output file name for the segment
        sf.write(out_filename, block, sr) # Save the segment
        counter += 1 # Increment the counter
        samples_written += buffer # Update the number of samples written

    st.write(f"Audio split into {counter-1} segments.") # Display the number of segments

    return split_dir

# Load speech-to-text model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/whisper-small"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=20,
    return_timestamps=True,
    torch_dtype=torch.float16,
    device=device,
)

# Function to perform productivity analysis
def evaluate_productivity(transcription, productivity_keywords):
    """
    Function to evaluate productivity of the meeting based on the transcription

    Parameters:
    transcription (str): Transcription of the meeting
    productivity_keywords (list): List of productivity keywords

    Returns:
    productive_segments (list): List of productive segments
    """
    # productivity_keywords = ["decision", "transcribe", "program", "task", "action", "plan", "solution", "agenda", "discuss"]
    productive_segments = [] # List to store productive segments
    sentences = sent_tokenize(transcription) # Tokenize the transcription into sentences
    for sentence in sentences: # Iterate over the sentences
        if any(keyword in sentence.lower() for keyword in productivity_keywords): # Check if the sentence contains any productivity keyword
            productive_segments.append(sentence) # Add the sentence to the list of productive segments
    return productive_segments

# Sentiment analysis function
def sentiment_analysis(text):
    """
    Function to perform sentiment analysis on the text

    Parameters:
    text (str): Text to perform sentiment analysis on

    Returns:
    sentiment (str): Sentiment of the text (POSITIVE, NEGATIVE, NEUTRAL)
    """
    sid = SentimentIntensityAnalyzer() # Initialize the sentiment analyzer
    sentiment_scores = sid.polarity_scores(text) # Get the sentiment scores
    sentiment = "POSITIVE" if sentiment_scores["compound"] >= 0.05 else "NEGATIVE" if sentiment_scores["compound"] <= -0.05 else "NEUTRAL" # Determine the sentiment based on the compound score
    return sentiment

# Streamlit UI
st.title("Meeting Analyzer") # Title of the app

# Upload audio file
audio_file = st.file_uploader("Upload your meeting audio file", type=["mp3", "wav", "flac"]) # File uploader for the audio file

if audio_file: # Proceed only if the audio file is uploaded
    st.audio(audio_file, format='audio/wav') # Display the uploaded audio file

    # Save the uploaded file
    with open("uploaded_audio.wav", "wb") as f: # Save the uploaded audio file
        f.write(audio_file.getbuffer())

    # Step 1: Preprocess the audio
    preprocessed_file = preprocess_audio("uploaded_audio.wav") # Preprocess the audio file
    st.write(f"Audio preprocessed")

    # Segment duration input box
    seg_dur = st.number_input("Enter the segment duration in seconds", value=30, min_value=1, max_value=60)

    # Step 2: Split the preprocessed audio
    split_dir = split_audio(preprocessed_file, segment_duration=seg_dur)

    # Step 3: Perform speech-to-text conversion
    st.header("Transcription:")
    transcription_ls = [] # List to store transcriptions
    audio_ls = sorted(os.listdir(split_dir)) # List of audio segments
    l = len(audio_ls)
    progress_bar = st.progress(0, text="") 
    for count, audio in enumerate(audio_ls, 1): # Iterate over the audio segments
        progress_bar.progress(count/l, text=f"Transcribing audio for segment #{count}...")
        result = pipe(os.path.join(split_dir, audio)) # Perform speech-to-text conversion
        transcription_ls.append(result["text"]) # Append the transcription to the list

    transcription = "\n\n".join(transcription_ls) # Join the transcriptions into a single string
    progress_bar.empty()
    st.text_area("", transcription, height=300) # Display the transcription

    # Step 4: Generate meeting minutes
    st.header("Meeting Minutes:")

    with st.spinner('Preparing Summary...'):
        message = "Give point-wise minutes of the meeting from this text:\n\n" + transcription # Prepare the message for the chatbot

        final_note: ChatResponse = chat(model="mistral", messages=[{"role": "user", "content": message}]) # Generate the meeting minutes
    st.text_area("", final_note.message.content, height=300) # Display the meeting minutes

    # Step 5: Sentiment analysis
    st.header("Sentiment Analysis:")
    with st.spinner('Evaluating Overall Sentiment...'):
        sentiment = sentiment_analysis(transcription) # Perform sentiment analysis on the transcription
    
    st.write(f"Overall Meeting Sentiment: {sentiment}") # Display the overall sentiment

    sentiment_per_segment = [] # List to store sentiment per segment
    progress_bar = st.progress(0, text="")
    for count, segment in enumerate(transcription_ls, 1): # Iterate over the segments for sentiment analysis
        progress_bar.progress(count/l, text=f"Analyzing sentiment for segment #{count}...")
        sentiment_per_segment.append(f"Sentiment for Segment#{count}: {sentiment_analysis(segment)}\n") # Perform sentiment analysis on each segment

    progress_bar.empty()
    st.write("\n".join(sentiment_per_segment)) # Display the sentiment per segment

    # Step 6: Evaluate productivity
    st.header("Productivity:")
    productivity_keywords = st.text_input("Enter productivity keywords separated by commas", "decision, transcribe, program, task, action, plan, solution, agenda, discuss") # Input box for productivity keywords
    productivity_keywords = productivity_keywords.split(", ") # Split the keywords by commas
    if productivity_keywords: # Proceed only if the keywords are entered
        productive_segments = evaluate_productivity(transcription, productivity_keywords) # Evaluate productivity based on the transcription
        count = len(productive_segments)
        for idx, segment in enumerate(productive_segments, 1): # Iterate over the productive segments
            st.write(f"Productive Segment {idx}: {segment}") # Display the productive segments