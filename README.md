# babelfish
Proof of concept python script that listens to an audio stream and translates any speech into English using a realistic-sounding voice. Uses OpenAI's whisper model for transcription and translation and TTS for voices. Translate YouTube videos, Zoom calls, or any audio into English in near real-time.

## Steps to install and run

These steps work on a Mac, YMMV on other platforms.

You must have a paid OpenAI API key to use it.

  1. Install blackhole (https://github.com/ExistentialAudio/BlackHole). Blackhole 2ch works best.
  1. Install dependencies: `pip install openai scipy sounddevice soundfile`
  1. (Optional) Install whisper for local translate: `pip install whisper`
  1. Click on Sound icon in menu bar and set audio output to `Blackhole 2ch`.
  1. List devices: `python3 -m sounddevice`
  1. Run babelfish `OPENAI_API_KEY=sk-..... python3 babelfish.py --input_device <Blackhole 2ch Device Number> --output_device <Output Device Number>`

## Notes

Supports local translation using whisper local model using the --local_translation flag. 
On my M1 Macbook Pro I wasn't able to run it locally in real-time with an accurate enough model. 
You may have better luck with an M2 or M3. 

Example: 

`OPENAI_API_KEY=sk-..... python3 babelfish.py --input_device <Blackhole 2ch Device Number> --output_device <Output Device Number> --local_translate True --local_translate_model large-v3`

Latency could be greatly improved, this is just a proof-of-concept.
Ideally it would be smart enough to only cut off the recordings between words.

`play_file.py` is included here as a simple way to test outputting an audio file to a device. This can be used to test babelfish by sending an audio file to the Blackhole 2ch device while babelfish is listening.

