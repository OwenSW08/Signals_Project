# this can be run locally if you install the dependencies, but to run it here
# there's issues you have to solve with sounddevice's dependencies
import tkinter as tk
from tkinter import messagebox
from typing import TypedDict, Callable

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

"""Signal in time domain. samplerate is the number of samples per second."""
class T_Signal(TypedDict):
    signal: np.ndarray
    samplerate: int

"""Signal in frequency domain. delta_w is the difference in frequency between two consecutive entries."""
class W_Signal(TypedDict):
    signal: np.ndarray
    delta_w: float

"""Generate transfer function in W_signal format from a callback function (in form f(w)=H(jw)), given list of frequencies. Note that for delta_w in the output is  only accurate if the input frequencies are evenly spaced."""
def get_fit_transfer_func(transfer_func: Callable[[float], complex], frequencies: np.ndarray) -> W_Signal:
    transfer_func_vals = np.array([transfer_func(w) for w in frequencies])
    delta_w = frequencies[1] - frequencies[0]

    return { "signal": transfer_func_vals, "delta_w": delta_w }

"""Match array-based transfer function to calculated frequencies of some fft, so that output[i] = transfer_func[fft_freqs[i] / transfer_func["delta_w"]]."""
def fit_array_transfer_func(transfer_func: W_Signal, fft_freqs: np.ndarray) -> W_Signal:
    # fit transfer function to fft frequencies
    new_transfer_func = np.zeros(len(fft_freqs))

    for i in range(len(fft_freqs)):
        # get nearest 2 frequencies in transfer function, and interpolate
        low_index = fft_freqs[i] // transfer_func["delta_w"]
        high_index = low_index + 1

        if high_index < len(transfer_func["signal"]):
            partition = (fft_freqs[i] / transfer_func["delta_w"]) % 1
            new_transfer_func[i] = transfer_func["signal"][low_index] * (1 - partition) + transfer_func["signal"][high_index] * partition
        else:
            new_transfer_func[i] = transfer_func["signal"][low_index]


    return { "signal": new_transfer_func, "delta_w": fft_freqs[1] - fft_freqs[0] }

"""Generate impulse response from callback function (in form f(t)=h(t)), given a sampling rate and length."""
def get_fit_impulse_response(impulse_response: Callable[[float], float], samplerate: int, length: float) -> T_Signal:
    impulse_response_vals = np.array([impulse_response(t) for t in np.arange(0, length, 1 / samplerate)])

    return { "signal": impulse_response_vals, "samplerate": samplerate }

"""Return given array-based impulse response converted to given sampling rate. If sampling rate is the same, return copy."""
def fit_array_impulse_response(impulse_response: T_Signal, samplerate: int) -> T_Signal:
    if impulse_response["samplerate"] == samplerate:
        return { "signal": impulse_response["signal"].copy(), "samplerate": samplerate }

    new_impulse_response = np.zeros(int(len(impulse_response["signal"]) * impulse_response["samplerate"] / samplerate))

    for i in range(len(new_impulse_response)):
        low_index = i * samplerate // impulse_response["samplerate"]
        high_index = low_index + 1

        if high_index < len(impulse_response["signal"]):
            partition = (i * samplerate / impulse_response["samplerate"]) % 1
            new_impulse_response[i] = impulse_response["signal"][low_index] * (1 - partition) + impulse_response["signal"][high_index] * partition
        else:
            new_impulse_response[i] = impulse_response["signal"][low_index]

    return { "signal": new_impulse_response, "samplerate": samplerate }

class AudioRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder")

        self.recording = False
        self.audio_data: T_Signal = { "signal": np.array([]), "samplerate": 1 } # audio data recorded from microphone

        self.filtered_audio_data: T_Signal = { "signal": np.array([]), "samplerate": 1 } # audio data after applying filter


        self.record_button = tk.Button(root, text="Record", command=self.record_audio)
        self.record_button.pack(pady=10)

        self.play_button = tk.Button(root, text="Play", command=self.play_audio)
        self.play_button.pack(pady=10)

        self.save_button = tk.Button(root, text="Save as .wav", command=self.save_audio)
        self.save_button.pack(pady=10)

    def record_audio(self):
        if not self.recording:
            self.recording = True
            self.record_button.config(text="Stop Recording")

            samplerate = 44100
            channels = 2

            audio_data = sd.rec(int(5 * 44100), samplerate=samplerate, channels=channels, dtype='int16')

            self.audio_data = { "signal": audio_data[0:len(audio_data):channels], "samplerate": samplerate } # to reduce to one time-domain signal, only take one channel

            sd.wait()
            self.recording = False
            self.record_button.config(text="Record")
            messagebox.showinfo("Info", "Recording finished")
        else:
            messagebox.showwarning("Warning", "Already recording")

    def play_audio(self):
        if self.audio_data is not None:
            sd.play(self.audio_data, samplerate=44100)
            sd.wait()
        else:
            messagebox.showwarning("Warning", "No audio recorded")

    def save_audio(self):
        if self.audio_data is not None:
            write('output.wav', 44100, self.audio_data)
            messagebox.showinfo("Info", "Audio saved as output.wav")
        else:
            messagebox.showwarning("Warning", "No audio recorded")

     # TODO: decompose these into frequency and time domain functions
    """Apply array-based transfer function to the fourier transform of audio data, convolve audio data with impulse response (might remove, since this the transfer function can accomplish this), compress final time-domain signal by factor (multiply sampling rate) and return filtered audio data."""
    """
    def apply_filter_type1(self, audio_signal: T_Signal, transfer_func: W_Signal, factor: float = 1, impulse_response: T_Signal | None = None) -> T_Signal:

        # note: for efficiency, we might try setting n to a power of 2 in the future
        fft_audio_signal = np.fft.fft(audio_signal["signal"])
        fft_freqs = np.fft.fftfreq(len(fft_audio_signal), d = 1 / audio_signal["samplerate"])

        # apply transfer function to fourier transform of audio signal
        fit_transfer_func = fit_array_transfer_func(transfer_func, fft_freqs)
        fft_filtered_audio_signal = fft_audio_signal * fit_transfer_func["signal"]

        # convert back to time domain
        filtered_audio_signal = np.fft.ifft(fft_filtered_audio_signal).real

        # convolve with impulse response, if given
        if (impulse_response is not None):
            fit_impulse_response = fit_array_impulse_response(impulse_response, audio_signal["samplerate"])
            final_filtered_audio_signal = np.convolve(filtered_audio_signal, fit_impulse_response["signal"], mode='same')

            return { "signal": final_filtered_audio_signal, "samplerate": int(audio_signal["samplerate"] * factor) }
        else:
            return { "signal": filtered_audio_signal, "samplerate": int(audio_signal["samplerate"] * factor) }
    """
    """Apply array-based transfer function to the fourier transform of audio data, convolve audio data with impulse response (might remove, since this the transfer function can accomplish this), compress final time-domain signal by factor (multiply sampling rate) and return filtered audio data."""
    """def apply_filter_type2(self, audio_signal: T_Signal, transfer_func: Callable[[float], complex], factor: float = 1, impulse_response: Callable[[float], float] | None = None, impulse_length: int = 100) -> T_Signal:

        # note: for efficiency, we might try setting n to a power of 2 in the future
        fft_audio_signal = np.fft.fft(audio_signal["signal"])
        fft_freqs = np.fft.fftfreq(len(fft_audio_signal), d = 1 / audio_signal["samplerate"])

        # apply transfer function to fourier transform of audio signal
        fit_transfer_func = get_fit_transfer_func(transfer_func, fft_freqs)
        fft_filtered_audio_signal = fft_audio_signal * fit_transfer_func["signal"]

        # convert back to time domain
        filtered_audio_signal = np.fft.ifft(fft_filtered_audio_signal).real

        # convolve with impulse response, if given
        if (impulse_response is not None):
            fit_impulse_response = get_fit_impulse_response(impulse_response, audio_signal["samplerate"], impulse_length)
            final_filtered_audio_signal = np.convolve(filtered_audio_signal, fit_impulse_response["signal"], mode='same')

            return { "signal": final_filtered_audio_signal, "samplerate": int(audio_signal["samplerate"] * factor) }
        else:
            return { "signal": filtered_audio_signal, "samplerate": int(audio_signal["samplerate"] * factor) }
    """

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorder(root)
    root.mainloop()