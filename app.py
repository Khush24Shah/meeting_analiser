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
    audio, sr = librosa.load(file_name, mono=True)
    audio = audio / np.max(np.abs(audio))  # Normalize the audio
    audio_denoised = medfilt(audio, kernel_size=3)  # Basic noise reduction
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    preprocessed_file = os.path.join(output_dir, f"preprocessed_{os.path.basename(file_name)}")
    sf.write(preprocessed_file, audio_denoised, sr)
    return preprocessed_file

# Function to split audio into segments
def split_audio(file_name, segment_duration, output_dir="out"):
    audio, sr = librosa.load(file_name)
    # audio = librosa.effects.time_stretch(audio, rate=0.95)  # Slow down the audio
    buffer = segment_duration * sr
    samples_total = len(audio)
    samples_written = 0
    counter = 1
    file_base_name = os.path.splitext(os.path.basename(file_name))[0]
    split_dir = os.path.join(output_dir, file_base_name)
    
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    while samples_written < samples_total:
        buffer = min(samples_total - samples_written, buffer)

        block = audio[samples_written: (samples_written + buffer)]
        out_filename = os.path.join(split_dir, f"split_{counter}.wav")
        sf.write(out_filename, block, sr)
        counter += 1
        samples_written += buffer

    st.write(f"Audio split into {counter-1} segments.")

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
def evaluate_productivity(transcription, productivity_keywords=None):
    # productivity_keywords = ["decision", "transcribe", "program", "task", "action", "plan", "solution", "agenda", "discuss"]
    productive_segments = []
    sentences = sent_tokenize(transcription)
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in productivity_keywords):
            productive_segments.append(sentence)
    return productive_segments

# Sentiment analysis function
def sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    sentiment = "POSITIVE" if sentiment_scores["compound"] >= 0.05 else "NEGATIVE" if sentiment_scores["compound"] <= -0.05 else "NEUTRAL"
    return sentiment

# Streamlit UI
st.title("Meeting Analyzer")

# Upload audio file
audio_file = st.file_uploader("Upload your meeting audio file", type=["mp3", "wav", "flac"])

if audio_file:
    st.audio(audio_file, format='audio/wav')

    # Save the uploaded file
    with open("uploaded_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    # Step 1: Preprocess the audio
    preprocessed_file = preprocess_audio("uploaded_audio.wav")
    st.write(f"Audio preprocessed")

    seg_dur = st.number_input("Enter the segment duration in seconds", value=30, min_value=1, max_value=60)

    # Step 2: Split the preprocessed audio
    split_dir = split_audio(preprocessed_file, segment_duration=seg_dur)

    # Step 3: Perform speech-to-text conversion
    transcription_ls = []
    audio_ls = sorted(os.listdir(split_dir))
    for audio in audio_ls:
        result = pipe(os.path.join(split_dir, audio))
        transcription_ls.append(result["text"])

    transcription = "\n\n".join(transcription_ls)
    st.subheader("Transcription:")
    st.text_area("Transcription", transcription, height=300)

    # Step 4: Generate meeting minutes
    message = "Give point-wise minutes of the meeting from this text:\n\n" + transcription

    final_note: ChatResponse = chat(model="mistral", messages=[{"role": "user", "content": message}])
    st.subheader("Meeting Minutes 1:")
    st.text_area("Minutes", final_note.message.content, height=300)

    # Step 4: Generate meeting minutes
    llm = Ollama(model="mistral")
    final_note = llm.invoke(f"Give point-wise minutes of the meeting from this text: {transcription}")
    st.subheader("Meeting Minutes 2:")
    st.text_area("Minutes", final_note, height=300)

    # Step 5: Sentiment analysis
    sentiment = sentiment_analysis(transcription)
    st.subheader("Sentiment Analysis:")
    st.write(f"Overall Meeting Sentiment: {sentiment}")

    sentiment_per_segment = [f"Sentiment for Segment#{count}: {sentiment_analysis(segment)}\n" for count, segment in enumerate(transcription_ls, 1)]
    st.write("\n".join(sentiment_per_segment))

    # Step 6: Evaluate productivity
    st.subheader("Productivity:")
    productivity_keywords = st.text_input("Enter productivity keywords separated by commas", "decision, transcribe, program, task, action, plan, solution, agenda, discuss")
    productivity_keywords = productivity_keywords.split(", ")
    if productivity_keywords:
        productive_segments = evaluate_productivity(transcription, productivity_keywords)
        count = len(productive_segments)
        for idx, segment in enumerate(productive_segments, 1):
            st.write(f"Productive Segment {idx}: {segment}")