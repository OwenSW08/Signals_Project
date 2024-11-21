from __future__ import annotations

import tkinter as tk
import wave
from tkinter import messagebox
from typing import TypedDict, Callable
import numpy
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write


"""Signal in time domain. samplerate is the number of samples per second."""

class T_Signal(TypedDict):
    signal: np.ndarray
    samplerate: int


class AudioRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder")

        self.recording = False
        self.audio_data: T_Signal = {"signal": np.array([]), "samplerate": 1}  # audio data recorded from microphone

        self.filtered_audio_data: T_Signal = {"signal": np.array([]),
                                              "samplerate": 1}  # audio data after applying filter

        self.record_button = tk.Button(root, text="Record", command=self.record_audio)
        self.record_button.pack(pady=10)

        self.play_button = tk.Button(root, text="Play", command=self.play_audio)
        self.play_button.pack(pady=10)

        self.save_button = tk.Button(root, text="Save as .wav", command=self.save_audio)
        self.save_button.pack(pady=10)
        #TODO change button names
        self.filter1_button = tk.Button(root, text="demo_filter1", command=self.demo_filter1)
        self.filter1_button.pack(pady=10)

        self.filter2_button = tk.Button(root, text="demo_filter2", command=self.demo_filter2)
        self.filter2_button.pack(pady=10)

        self.play_filtered_audio_button = tk.Button(root, text="Play filtered audio", command=self.play_filtered_audio)
        self.play_filtered_audio_button.pack(pady=10)

    def record_audio(self):
        if not self.recording:
            self.recording = True
            self.record_button.config(text="Stop Recording")

            samplerate = 44100
            channels = 1  # alert: there are things that won't work if this gets changed, I think

            audio_data = sd.rec(int(5 * 44100), samplerate=samplerate, channels=channels)

            self.audio_data = {"signal": audio_data[0:len(audio_data):channels],
                               "samplerate": samplerate}  # to reduce to one time-domain signal, only take one channel

            sd.wait()
            self.recording = False
            self.record_button.config(text="Record")
            messagebox.showinfo("Info", "Recording finished")
        else:
            messagebox.showwarning("Warning", "Already recording")

    def play_audio(self):
        if self.audio_data is not None:
            sd.play(self.audio_data["signal"], samplerate=self.audio_data["samplerate"])
            sd.wait()
        else:
            messagebox.showwarning("Warning", "No audio recorded")

    def save_audio(self):
        if self.audio_data is not None:
            write('output.wav', self.audio_data["samplerate"], self.audio_data["signal"])
            messagebox.showinfo("Info", "Audio saved as output.wav")
        else:
            messagebox.showwarning("Warning", "No audio recorded")

    def play_filtered_audio(self):
        if self.filtered_audio_data is not None:
            sd.play(self.filtered_audio_data["signal"], samplerate=self.filtered_audio_data["samplerate"])
            sd.wait()
        else:
            messagebox.showwarning("Warning", "No audio recorded")
    def demo_filter1(self):
        self.filtered_audio_data = self.audio_data

    def demo_filter2(self):
        self.filtered_audio_data = self.audio_data

    def ghost_filter(self):
        sound = self.audio_data["signal"]
        sound.set_reverse()
        sound.set_echo(0.05)
        sound.set_reverse()
        sound.set_audio_speed(.70)
        sound.set_audio_pitch(2)
        sound.set_volume(8.0)
        sound.set_bandpass(50, 3200)


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorder(root)
    root.mainloop()
