#!/usr/bin/env python3

import librosa
import numpy as np
import pyfiglet
import queue
import sounddevice as sd
import time
import whisper

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

def main():
    stt = whisper.load_model('base.en')

    input_audio = record_x_seconds(5)
    # sd.play(input_audio, mapping=[1])

    stt_audio = librosa.resample(input_audio, orig_sr=44100, target_sr=16000)

    empty_text = stt.transcribe(input_audio, fp16=False)

    if len(empty_text['text']) > 3:
        pyfiglet.print_figlet(empty_text['text'], font='clb8x10')
    else:
        print(empty_text)
        print(empty_text['text'])

if __name__ == "__main__":
    main()
