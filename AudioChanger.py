import sys, wave
import numpy as np
from numpy import array, int16
from scipy.signal import lfilter, butter
from scipy.io.wavfile import read, write
from scipy import signal
import T_Signal

class AudioChanger(object):
    __slots__ = ('audio_data', 'sample_freq')

    def __init__(self, input_audio_path: T_Signal.T_Signal):

        self.sample_freq = input_audio_path["samplerate"]
        self.audio_data = input_audio_path["signal"]
        self.audio_data = AudioChanger.normalize(self.audio_data)

    @staticmethod
    def normalize (audio):
        """Divide audio data by its maximum value to normalize the data"""
        normalized_signal = (audio / np.max(np.abs(audio)))
        return normalized_signal

    def set_audio_speed(self, speed_factor):
        '''Multiply the speed of the audio by a floating-point factor'''
        sound_index = np.round(np.arange(0, len(self.audio_data), speed_factor))
        self.audio_data = self.audio_data[sound_index[sound_index < len(self.audio_data)].astype(int)]

    def set_echo(self, delay):
        '''Applies an echo that is 0...<input audio duration in seconds> seconds from the beginning'''
        output_audio = np.zeros(len(self.audio_data))
        output_delay = delay * self.sample_freq

        for count, e in enumerate(self.audio_data):
            output_audio[count] = e + self.audio_data[count - int(output_delay)]

        self.audio_data = output_audio

    def set_volume(self, level):
        '''Sets the overall volume of the data via floating-point factor (multiply volume by level)'''
        output_audio = np.zeros(len(self.audio_data))

        for count, e in enumerate(self.audio_data):
            output_audio[count] = (e * level)

        self.audio_data = output_audio

    def set_audio_pitch(self, n, window_size=2 ** 13, h=2 ** 11):
        '''Sets the pitch of the audio to a certain threshold (speed up audio by factor to increase the frequecy, but then stretch while maintaining frequency to restore the original speed)'''
        factor = 2 ** (1.0 * n / 12.0)
        self._set_stretch(1.0 / factor, window_size, h)
        self.audio_data = self.audio_data[window_size:]
        self.set_audio_speed(factor)

    def _set_stretch(self, factor, window_size, h):
        """Stretch audio length by factor while maintaining pitch using the Phase Vocoder technique"""
        phase = np.zeros(window_size)
        hanning_window = np.hanning(window_size)
        result = np.zeros(int(len(self.audio_data) / factor + window_size))

        for i in np.arange(0, len(self.audio_data) - (window_size + h), h * factor):
            # Two potentially overlapping subarrays
            a1 = self.audio_data[int(i): int(i + window_size)]
            a2 = self.audio_data[int(i + h): int(i + window_size + h)]

            # The spectra of these arrays, with the hanning window applied (to smooth the signal and stop leakage)
            s1 = np.fft.fft(hanning_window * a1)
            s2 = np.fft.fft(hanning_window * a2)

            # Rephase all frequencies
            phase = (phase + np.angle(s2 / s1)) % 2 * np.pi

            a2_rephased = np.fft.ifft(np.abs(s2) * np.exp(1j * phase))
            i2 = int(i / factor)
            result[i2: i2 + window_size] += hanning_window * a2_rephased.real

        # normalize (16bit)
        result = ((2 ** (16 - 4)) * result / result.max())
        self.audio_data = result.astype('int16')

    def set_lowpass(self, cutoff_low, order=5):
        '''Applies a low pass filter'''
        nyquist = self.sample_freq # shouldn't there be a *2 here?
        cutoff = cutoff_low / nyquist # this is the "normalized" frequency - signal.butter's cutoff input takes a float from 0 to 1, where 1 corresponds to the Nyquist frequency (so the unit of this is half cycles/sample)
        x, y = signal.butter(order, cutoff, btype='lowpass', analog=False) # consider adding output='sos', to avoid floating-point errors associated with higher-order polynomial to root relationships
        self.audio_data = signal.filtfilt(x, y, self.audio_data)

    def set_highpass(self, cutoff_high, order=5):
        '''Applies a high pass filter'''
        nyquist = self.sample_freq  # repeat: shouldn't there be a *2 here?
        cutoff = cutoff_high / nyquist # this is the "normalized" frequency - signal.butter's cutoff input takes a float from 0 to 1, where 1 corresponds to the Nyquist frequency (so the unit of this is half cycles/sample)
        x, y = signal.butter(order, cutoff, btype='highpass', analog=False) # consider adding output='sos'
        self.audio_data = signal.filtfilt(x, y, self.audio_data)

    def set_bandpass(self, cutoff_low, cutoff_high, order=5):
        '''Applies a band pass filter'''
        cutoff = np.zeros(2)
        nyquist = self.sample_freq # repeat: shouldn't there be a *2 here?

        # these are the "normalized" frequencies - signal.butter's cutoff input takes floats from 0 to 1, where 1 corresponds to the Nyquist frequency (so the unit of this is half cycles/sample)
        cutoff[0] = cutoff_low / nyquist
        cutoff[1] = cutoff_high / nyquist
        x, y = signal.butter(order, cutoff, btype='bandpass', analog=False) # consider adding output='sos'
        self.audio_data = signal.filtfilt(x, y, self.audio_data)

    def get_audio_data(self):
        return T_Signal.T_Signal(signal = self.audio_data, samplerate = self.sample_freq)
