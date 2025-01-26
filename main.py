#!/usr/bin/env python3

import json
import librosa
import numpy as np
import pyfiglet
import queue
import requests
import sounddevice as sd
import time
import whisper

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
              'n_predict': 128}

    result = requests.post('http://localhost:8080/completion',
                           json.dumps(prompt))

    return result.json()['content']

def main():
    stt = whisper.load_model('base.en')

    input_audio = record_x_seconds(5)
    # sd.play(input_audio, mapping=[1])

    stt_audio = librosa.resample(input_audio, orig_sr=44100, target_sr=16000)

    transcribed_text = stt.transcribe(input_audio, fp16=False)['text']

    print(transcribed_text)

    answer_text = ask_ai(transcribed_text)


    print("\n\n\n")

    print(answer_text)

if __name__ == "__main__":
    main()
