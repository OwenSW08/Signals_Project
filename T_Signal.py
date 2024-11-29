from typing import TypedDict

import numpy as np

"""Signal in time domain. samplerate is the number of samples per second."""

class T_Signal(TypedDict):
    signal: np.ndarray
    samplerate: int
