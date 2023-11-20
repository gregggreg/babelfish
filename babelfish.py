#!/usr/bin/env python3
"""Records audio, translates it to English, and speaks the text 
using a realistic-sounding human voice.

openai, scipy, sounddevice, soundfile modules must be installed for this to work.
whisper module must be installed for local translation.

set OPENAI_API_KEY environment variable to your API key.

Use `python3 -m sounddevice` to list devices.

"""
import argparse
import io
import os
import queue
import threading
import whisper
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from openai import OpenAI
from scipy.io.wavfile import write

parser = argparse.ArgumentParser()
parser.add_argument('--input_device', type=int, default=1, help='Input device ID. Use `python3 -m sounddevice` to list devices.')
parser.add_argument('--output_device', type=int, default=3, help='Output device ID. Use `python3 -m sounddevice` to list devices.')
parser.add_argument('--voice', type=str, default="alloy", help='Voice to use for translation. Options: alloy, echo, fable, onyx, nova, or shimmer')
parser.add_argument('--local_translate', type=bool, default=False, help='Translate with local whisper model.')
parser.add_argument('--local_translate_model', type=str, default="large-v3", help='Local whisper model to use.')

args = parser.parse_args()

input_device = args.input_device
output_device = args.output_device
local_translate = args.local_translate
local_translate_model = args.local_translate_model
voice = args.voice

client = OpenAI()
if local_translate:
    model = whisper.load_model(local_translate_model)
current_frame = 0
translate_queue = queue.Queue()

def worker():
    while True:
        item = translate_queue.get()
        if item is None:
            break
        filename = item
        if local_translate:
            translate_audio_local(filename)
        else:
            translate_audio_remote(filename)
        translate_queue.task_done()

worker_thread = threading.Thread(target=worker)
worker_thread.start()

def write_audio(data, samplerate, filename):
    write(filename, samplerate, data)
    translate_queue.put(filename)

def load_audio():
    print("Recording audio...")
    fs = 16000  # Sample rate
    seconds = 3  # Duration of recording
    uuid = os.urandom(16).hex()
    filename = "/tmp/" + uuid + ".wav"
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, device=input_device)
    sd.wait()  # Wait until recording is finished
    write_thread = threading.Thread(target=write_audio, args=(myrecording,fs,filename,))
    write_thread.start()
    return filename

def output_audio(text):
    global current_frame
    current_frame = 0
 
    if len(text) > 0:
        response = client.audio.speech.create(
	        model="tts-1",
	        voice=voice,
	        input=text
	    )
        data, samplerate = sf.read(io.BytesIO(response.content), always_2d=True)

        def callback(outdata, frames, time, status):
            global current_frame
            if status:
                print(status)
            chunksize = min(len(data) - current_frame, frames)
            outdata[:chunksize] = data[current_frame:current_frame + chunksize]
            if chunksize < frames:
                outdata[chunksize:] = 0
                raise sd.CallbackStop()
            current_frame += chunksize

        def stream_audio(samplerate, channels, callback):
            event = threading.Event()
            stream = sd.OutputStream(samplerate=samplerate, channels=channels, device=output_device, 
                                     callback=callback, finished_callback=event.set)
            with stream:
                event.wait()  # Keep waiting for the stream to finish

        stream_audio(samplerate, data.shape[1], callback)
    else:
        print("(No text to translate)")

def translate_audio_remote(filename):
    print("Transcribing audio remotely...")
    text = client.audio.translations.create(
        model="whisper-1", 
        file=Path(filename),
        response_format="text"
    )
    print(text)
    output_audio(text)
    os.remove(filename)

def translate_audio_local(filename):   
    print("Transcribing audio locally...")
    response = model.transcribe(filename, task="translate")
    text = response["text"]
    print(text)
    output_audio(text)
    os.remove(filename)

while True:
    load_audio()

