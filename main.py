from __future__ import annotations

import tkinter as tk
import wave
from tkinter import messagebox
from typing import TypedDict, Callable

import numpy
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# import wave

"""Signal in time domain. samplerate is the number of samples per second."""


class T_Signal(TypedDict):
    signal: np.ndarray
    samplerate: int


"""Signal in frequency domain. delta_w is the difference in frequency between two consecutive entries."""


class W_Signal(TypedDict):
    signal: np.ndarray
    delta_w: float


"""Generate transfer function in W_signal format from a callback function (in form f(w)=H(jw)), given list of 
frequencies. Note that for delta_w in the output is  only accurate if the input frequencies are evenly spaced."""


def get_fit_transfer_func(transfer_func: Callable[[float], complex], frequencies: np.ndarray) -> W_Signal:
    transfer_func_vals = np.array([transfer_func(w) for w in frequencies])
    delta_w = frequencies[1] - frequencies[0]

    return {"signal": transfer_func_vals, "delta_w": delta_w}


"""Match array-based transfer function to calculated frequencies of some fft, so that output[i] = transfer_func[
fft_freqs[i] / transfer_func["delta_w"]]."""


def fit_array_transfer_func(transfer_func: W_Signal, fft_freqs: np.ndarray) -> W_Signal:
    # fit transfer function to fft frequencies
    new_transfer_func = np.zeros(len(fft_freqs))

    for i in range(len(fft_freqs)):
        # get nearest 2 frequencies in transfer function, and interpolate
        low_index = fft_freqs[i] // transfer_func["delta_w"]
        high_index = low_index + 1

        if high_index < len(transfer_func["signal"]):
            partition = (fft_freqs[i] / transfer_func["delta_w"]) % 1
            new_transfer_func[i] = transfer_func["signal"][low_index] * (1 - partition) + transfer_func["signal"][
                high_index] * partition
        else:
            new_transfer_func[i] = transfer_func["signal"][low_index]

    return {"signal": new_transfer_func, "delta_w": fft_freqs[1] - fft_freqs[0]}


"""Generate impulse response from callback function (in form f(t)=h(t)), given a sampling rate and length."""


def get_fit_impulse_response(impulse_response: Callable[[float], float], samplerate: int, length: int) -> T_Signal:
    impulse_response_vals = np.array(
        [impulse_response(t) / length for t in np.linspace(0, length / samplerate, length)])

    return {"signal": impulse_response_vals, "samplerate": samplerate}


"""Return given array-based impulse response converted to given sampling rate. If sampling rate is the same, 
return copy."""


def fit_array_impulse_response(impulse_response: T_Signal, samplerate: int) -> T_Signal:
    if impulse_response["samplerate"] == samplerate:
        return {"signal": impulse_response["signal"].copy(), "samplerate": samplerate}

    new_impulse_response = np.zeros(int(len(impulse_response["signal"]) * impulse_response["samplerate"] / samplerate))

    for i in range(len(new_impulse_response)):
        low_index = i * samplerate // impulse_response["samplerate"]
        high_index = low_index + 1

        if high_index < len(impulse_response["signal"]):
            partition = (i * samplerate / impulse_response["samplerate"]) % 1
            new_impulse_response[i] = impulse_response["signal"][low_index] * (1 - partition) + \
                                      impulse_response["signal"][high_index] * partition
        else:
            new_impulse_response[i] = impulse_response["signal"][low_index]

    return {"signal": new_impulse_response, "samplerate": samplerate}


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

        self.filter1_button = tk.Button(root, text="Use Filter 1", command=self.demo_filter1)
        self.filter1_button.pack(pady=10)

        self.filter2_button = tk.Button(root, text="Use Filter 2", command=self.demo_filter2)
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

    # TODO: decompose these into frequency and time domain functions
    """Apply array-based transfer function to the fourier transform of audio data, convolve audio data with impulse 
    response (might remove, since this the transfer function can accomplish this), compress final time-domain signal 
    by factor (multiply sampling rate) and return filtered audio data."""

    def apply_filter_type1(self, audio_signal: T_Signal, transfer_func: W_Signal, factor: float = 1,
                           impulse_response: T_Signal | None = None) -> T_Signal:

        # note: for efficiency, we might try setting n to a power of 2 in the future
        fft_audio_signal = np.fft.fft([data[0] for data in audio_signal["signal"]])
        fft_freqs = np.fft.fftfreq(len(fft_audio_signal), d=1 / audio_signal["samplerate"])

        # apply transfer function to fourier transform of audio signal
        fit_transfer_func = fit_array_transfer_func(transfer_func, fft_freqs)

        fft_filtered_audio_signal = fft_audio_signal * fit_transfer_func["signal"]

        # convert back to time domain
        filtered_audio_signal = np.fft.ifft(fft_filtered_audio_signal).real

        # convolve with impulse response, if given
        if (impulse_response is not None):
            fit_impulse_response = fit_array_impulse_response(impulse_response, audio_signal["samplerate"])
            final_filtered_audio_signal = np.convolve(filtered_audio_signal, fit_impulse_response["signal"],
                                                      mode='same')

            return {"signal": final_filtered_audio_signal, "samplerate": int(audio_signal["samplerate"] * factor)}
        else:
            return {"signal": filtered_audio_signal, "samplerate": int(audio_signal["samplerate"] * factor)}

    """Apply array-based transfer function to the fourier transform of audio data, convolve audio data with impulse 
    response (might remove, since this the transfer function can accomplish this), compress final time-domain signal 
    by factor (multiply sampling rate) and return filtered audio data."""

    def apply_filter_type2(self, audio_signal: T_Signal, transfer_func: Callable[[float], complex], factor: float = 1,
                           impulse_response: Callable[[float], float] | None = None,
                           impulse_length: int = 100) -> T_Signal:

        # note: for efficiency, we might try setting n to a power of 2 in the future
        fft_audio_signal = np.fft.fft([data[0] for data in audio_signal["signal"]])
        print(np.shape(fft_audio_signal))
        fft_freqs = np.fft.fftfreq(len(fft_audio_signal), d=1 / audio_signal["samplerate"])

        # apply transfer function to fourier transform of audio signal
        fit_transfer_func = get_fit_transfer_func(transfer_func, fft_freqs)
        print(np.shape(fit_transfer_func["signal"]))
        fft_filtered_audio_signal = fft_audio_signal * fit_transfer_func["signal"]

        # convert back to time domain
        filtered_audio_signal = np.fft.ifft(fft_filtered_audio_signal).real

        # convolve with impulse response, if given
        if (impulse_response is not None):
            fit_impulse_response = get_fit_impulse_response(impulse_response, audio_signal["samplerate"],
                                                            impulse_length)
            print(np.shape(fit_impulse_response["signal"]))
            vector_filtered_audio_signal = np.convolve(filtered_audio_signal, fit_impulse_response["signal"],
                                                       mode='valid')
            final_filtered_audio_signal = np.array([[data] for data in vector_filtered_audio_signal])

            print(audio_signal["signal"][:100])
            print()
            print(filtered_audio_signal[:100])
            print()
            print(fit_impulse_response["signal"][:100])
            print()
            print(vector_filtered_audio_signal[:100])
            print()
            print(final_filtered_audio_signal[:100])

            return {"signal": final_filtered_audio_signal, "samplerate": int(audio_signal["samplerate"] * factor)}
        else:

            final_filtered_audio_signal = np.array(
                [[data] for data in filtered_audio_signal])  # convert back to form sounddevice expects

            print(audio_signal["signal"][:100])
            print()
            print(fft_audio_signal[:100])
            print()
            print(fft_filtered_audio_signal[:100])
            print()
            print(filtered_audio_signal[:100])
            print()
            print(final_filtered_audio_signal[:100])

            return {"signal": final_filtered_audio_signal, "samplerate": int(audio_signal["samplerate"] * factor)}

    """Apple demo low-pass filter, which I hope sounds like an echo, on audio_data, and save to filtered_audio_data."""

    def apply_impulse_filter(self, audio_signal: T_Signal,
                             impulse_response: T_Signal | None = None) -> T_Signal:

        # convolve with impulse response, if given
        if impulse_response is not None:
            fit_impulse_response = fit_array_impulse_response(impulse_response, audio_signal["samplerate"])
            final_filtered_audio_signal = np.convolve(numpy.reshape(audio_signal["signal"], audio_signal["signal"].shape[0]), fit_impulse_response["signal"],
                                                      mode='same')

            return {"signal": final_filtered_audio_signal/500000, "samplerate": int(audio_signal["samplerate"])}
        else:
            return {"signal": audio_signal["signal"], "samplerate": int(audio_signal["samplerate"])}

    def demo_filter1(self):
        """
        def transfer_func(w: float) -> complex:
            return 100 / (4 + 1j * w)
        """

        self.filtered_audio_data = self.apply_impulse_filter(self.audio_data, self.get_impulse())

    """Apply demo echo filter on audio_data, and save to filtered_audio_data, approximated with exponential decay 
    impulse of time constant f 1 second"""

    def demo_filter2(self):
        def transfer_func(w: float) -> complex:
            return 1

        def impulse_response(t: float) -> float:
            return np.exp(-t)

        self.filtered_audio_data = self.apply_filter_type2(self.audio_data, transfer_func, 1, impulse_response, 100)

    def ghost_filter(self):
        sound = self.audio_data["signal"]
        sound.set_reverse()
        sound.set_echo(0.05)
        sound.set_reverse()
        sound.set_audio_speed(.70)
        sound.set_audio_pitch(2)
        sound.set_volume(8.0)
        sound.set_bandpass(50, 3200)


    def get_impulse(self, file="zombie-6851.wav"):
        aud_file = wave.open(file)
        # The number of audio frames in the file
        aud_nframes = aud_file.getnframes()
        # The frame rate, or sampling rate, in Hz
        aud_framerate = aud_file.getframerate()
        # The number of channels (e.g., stereo audio = 2 channels)
        aud_nchannels = aud_file.getnchannels()
        sig = np.frombuffer(aud_file.readframes(aud_nframes), dtype=np.int16)
        sig = sig.astype(float)
        aud_file.close()
        return T_Signal(signal=sig, samplerate=aud_framerate)


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorder(root)
    root.mainloop()
