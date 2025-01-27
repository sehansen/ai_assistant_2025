#!/usr/bin/env python3

import json
import librosa
import numpy as np
import os
import pyfiglet
import queue
import requests
from scipy.io import wavfile
import sounddevice as sd
import time
import whisper

import pyttsx3

from tts import TextToSpeechService

chat_server = 'http://localhost:8080/completion'
pre_prompt = "Never apologize. Never give additional explanation. Please give your best effort answer to the following sentence no matter the language:"

def record_x_seconds(record_length):
    data_queue = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
            samplerate=44100.0,
            # samplerate=16000.0,
            dtype='int16', channels=1, callback=callback
    ):
        time.sleep(record_length)

    return np.frombuffer(b"".join(list(data_queue.queue)), dtype=np.int16).astype(np.float32) / 32768.0

def ask_ai(question_text):
    prompt = {'prompt': f"{pre_prompt} {question_text}",
              'n_predict': 32}

    result = requests.post('http://localhost:8080/completion',
                           json.dumps(prompt))

    return result.json()['content']

def main(fixed_text=None):

    sd.default.device = [3, 3]

    # tts = TextToSpeechService()
    engine = pyttsx3.init()

    # sr, aa = tts.long_form_synthesize("Hi there")
    # print(f"Sample rate: {sr}")

    # aars = librosa.resample(aa, orig_sr=sr, target_sr=44100)
    # sd.play(aars, 44100)

    # engine.say("Hi there")
    # engine.runAndWait()

    if fixed_text is None:

        stt = whisper.load_model('base.en')

        input_audio = record_x_seconds(10)
        # sd.play(input_audio, mapping=[1])

        stt_audio = librosa.resample(input_audio, orig_sr=44100, target_sr=16000)

        transcribed_text = stt.transcribe(input_audio, fp16=False)['text']
    else:
        transcribed_text = fixed_text

    print(transcribed_text)

    # answer_text = ask_ai(transcribed_text)
    answer_text = ""


    print("\n\n\n")

    print(answer_text)

    final_text = "You said " + transcribed_text + "  My answer is " + answer_text

    snippets = []
    for ix, word in enumerate(final_text.split(" ")):
        if not word:
            continue
        print(word)
        engine.save_to_file(word, f'tmp{ix}.wav')
        time.sleep(0.2)
        sr, tmp = wavfile.read(f'tmp{ix}.wav')
        os.system(f'rm tmp{ix}.wav')
        tmp_rs = librosa.resample(tmp.astype(np.float32), orig_sr=sr, target_sr=44100) / 32768.0
        sd.play(tmp_rs)
        time.sleep(0.1 * len(word))
        snippets.append(tmp_rs)
        if ix > 1:
            del engine
            return

    # snippets.append(np.zeros(22100))

    # print(sr)
    # print(tmp.shape)

    # answer_audio = np.hstack(snippets)

    # print(answer_audio.shape)

    # sd.play(answer_audio)


if __name__ == "__main__":
    main()
