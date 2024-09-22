import pyaudio
import numpy as np
import threading
import queue
import tkinter as tk
import logging
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize speech recognition model
logger.info("Initializing speech recognition model")
asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

# Initialize text generation model
logger.info("Initializing text generation model")
generator = pipeline("text-generation", model="gpt2")

# Audio settings
CHUNK = 16000  # 1 second of audio at 16kHz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Create a queue to communicate between threads
audio_queue = queue.Queue()
text_queue = queue.Queue()

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def process_audio():
    logger.info("Starting audio processing")
    while True:
        audio_data = audio_queue.get()
        if audio_data is None:
            logger.info("Received signal to stop audio processing")
            break
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Process audio with speech recognition model
        logger.debug("Processing audio with speech recognition model")
        result = asr_pipeline(audio_np)
        transcribed_text = result["text"]
        text_queue.put(transcribed_text)
        logger.debug(f"Transcribed text: {transcribed_text}")

def generate_question(context):
    logger.info("Generating question")
    try:
        prompt = f"Based on the following context, generate a thought-provoking question:\n\nContext: {context}\n\nQuestion:"
        response = generator(prompt, max_length=50, num_return_sequences=1)
        question = response[0]['generated_text'].split("Question:")[-1].strip()
        logger.info(f"Generated question: {question}")
        return question
    except Exception as e:
        logger.error(f"Error generating question: {str(e)}")
        return "Error generating question. Please try again."

def update_display():
    context = ""
    while True:
        try:
            new_text = text_queue.get_nowait()
            context += new_text + " "
            context = context[-500:]  # Keep only the last 500 characters
            question = generate_question(context)
            question_label.config(text=question)
        except queue.Empty:
            pass
        root.after(100, update_display)

# Set up audio stream
logger.info("Setting up audio stream")
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback)

# Start audio processing thread
logger.info("Starting audio processing thread")
audio_thread = threading.Thread(target=process_audio)
audio_thread.start()

# Set up display
logger.info("Setting up display")
root = tk.Tk()
root.title("AI Thought Provoker")
question_label = tk.Label(root, text="Listening...", wraplength=400)
question_label.pack(padx=20, pady=20)

# Start updating display
root.after(100, update_display)

# Start audio stream
logger.info("Starting audio stream")
stream.start_stream()

# Run the GUI
logger.info("Running GUI")
try:
    root.mainloop()
except Exception as e:
    logger.error(f"Error in GUI: {str(e)}")

# Clean up
logger.info("Cleaning up")
stream.stop_stream()
stream.close()
p.terminate()
audio_queue.put(None)
audio_thread.join()
logger.info("Application closed")