from __future__ import annotations
import tkinter as tk
import wave
from tkinter import messagebox
from typing import TypedDict, Callable
import numpy
import sounddevice as sd
import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import write
from T_Signal import T_Signal
from AudioChanger import AudioChanger
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # (FigureCanvasTkAgg, NavigationToolbar2Tk)

class AudioRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder")
        self.root.geometry("800x500")
        self.main_box = tk.LabelFrame(root, pady=20)
        self.button_box = tk.LabelFrame(self.main_box, pady=20)
        self.main_box.pack(side="top", fill="x")
        self.button_box.pack(side="left", pady=20)
        fig = Figure(figsize=(6, 6), dpi=100)
        y = 0
        # adding the subplot
        plot1 = fig.add_subplot(111)
        # plotting the graph
        plot1.plot(y)
        plot1.set_title('Time Domain Graph of Audio Recording')
        plot1.set_xlabel('Samples')
        plot1.set_ylabel('Amplitude')
        self.canvas = FigureCanvasTkAgg(fig, master=self.main_box)
        self.canvas.draw()
        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().pack()

        self.recording = False
        self.audio_data: T_Signal = {"signal": np.array([]), "samplerate": 1}  # audio data recorded from microphone

        self.filtered_audio_data: T_Signal = {"signal": np.array([]),
                                              "samplerate": 1}  # audio data after applying filter

        self.record_button = tk.Button(self.button_box, text="Record", command=self.record_audio)
        self.record_button.pack(pady=10)

        self.play_button = tk.Button(self.button_box, text="Play", command=self.play_audio)
        self.play_button.pack(pady=10)

        self.filter1_button = tk.Button(self.button_box, text="Use Ghost Filter", command=self.ghost_filter)
        self.filter1_button.pack(pady=10)

        self.filter2_button = tk.Button(self.button_box, text="Use Demon Filter", command=self.demon_filter)
        self.filter2_button.pack(pady=10)

        self.filter3_button = tk.Button(self.button_box, text="Use Alien Filter", command=self.alien_filter)
        self.filter3_button.pack(pady=10)

        self.play_filtered_audio_button = tk.Button(self.button_box, text="Play Filtered Audio",
                                                    command=self.play_filtered_audio)
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
            self.plot(self.audio_data)
        else:
            messagebox.showwarning("Warning", "Already recording")

    def play_audio(self):
        if self.audio_data is not None:
            sd.play(self.audio_data["signal"], samplerate=self.audio_data["samplerate"])
            sd.wait()
        else:
            messagebox.showwarning("Warning", "No audio recorded")

    def play_filtered_audio(self):
        if self.filtered_audio_data is not None:
            sd.play(self.filtered_audio_data["signal"], samplerate=self.filtered_audio_data["samplerate"])
            sd.wait()
        else:
            messagebox.showwarning("Warning", "No audio recorded")

    def plot(self, audio_signal: T_Signal):
        self.canvas.get_tk_widget().destroy()
        fig = Figure(figsize=(6, 6), dpi=100)
        y = audio_signal["signal"]
        # adding the subplot
        plot1 = fig.add_subplot(111)
        # plotting the graph
        plot1.plot(y)
        plot1.set_title('Time Domain Graph of Audio Recording')
        plot1.set_xlabel('Samples')
        plot1.set_ylabel('Amplitude')
        self.canvas = FigureCanvasTkAgg(fig, master=self.main_box)
        self.canvas.draw()
        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().pack()

    def ghost_filter(self):
        sound = AudioChanger(self.audio_data)
        sound.set_audio_speed(0.7)
        sound.set_volume(5000)
        sound.set_echo(0.1)
        sound.set_echo(0.2)
        sound.set_highpass(1000)
        sound.set_audio_pitch(8)
        self.filtered_audio_data = sound.get_audio_data()
        self.plot(self.filtered_audio_data)

    def demon_filter(self):
        sound = AudioChanger(self.audio_data)
        sound.set_audio_speed(0.65)
        sound.set_volume(10)
        sound.set_echo(0.1)
        sound.set_lowpass(12000)
        self.filtered_audio_data = sound.get_audio_data()
        self.plot(self.filtered_audio_data)

    def alien_filter(self):
        sound = AudioChanger(self.audio_data)
        sound.set_volume(5)
        sound.set_echo(0.1)
        sound.set_audio_speed(1.2)
        sound.set_bandpass(4000, 30000)
        self.filtered_audio_data = sound.get_audio_data()
        self.plot(self.filtered_audio_data)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorder(root)
    root.mainloop()
