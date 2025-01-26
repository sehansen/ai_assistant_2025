#!/usr/bin/env python3

import numpy as np
import whisper

def main():
    stt = whisper.load_model('base.en')

    input_audio = np.zeros((100, 2), np.dtype('float32'))

    empty_text = stt.transcribe(input_audio, fp16=False)

    print(empty_text)

if __name__ == "__main__":
    main()
