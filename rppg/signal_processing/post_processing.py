import numpy as np
import scipy.signal as signal
from scipy import sparse

def detrend(input_signal, lambda_value = 100):
    if not isinstance(input_signal, np.ndarray):
        raise ValueError("Input signal must be a NumPy array")
    
    filtered_signal = signal.detrend(input_signal)
    return filtered_signal

def cheby2_bandpass_filter(input_signal, lowcut, highcut, fs, order=4):
    if not isinstance(input_signal, np.ndarray):
        raise ValueError("Input signal must be a NumPy array")
        
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.cheby2(order, 5, [low, high], btype="band")
    y = signal.filtfilt(b, a, input_signal)
    return y


def hilbert_filter(input_signal, lowcut, highcut, fs, order=4):
    if not isinstance(input_signal, np.ndarray):
        raise ValueError("Input signal must be a NumPy array")
    
    HilberFiltered = np.empty_like(input_signal, dtype=np.float64)

    for channel in range(3):
        analytic_signal = signal.hilbert(input_signal[:, channel]) 
        amplitude_envelope = np.abs(analytic_signal)
        HilberFiltered[:, channel] = input_signal[:, channel] / amplitude_envelope

    return HilberFiltered

def normalisation(input_signal):
    if not isinstance(input_signal, np.ndarray):
        raise ValueError("Input signal must be a NumPy array")

    normalized_signal = (input_signal - np.min(input_signal, axis=-1, keepdims=True)) / (np.max(input_signal, axis=-1, keepdims=True) - 
                                                                                         np.min(input_signal, axis=-1, keepdims=True))
    return normalized_signal

def standardisation(input_signal):
    if not isinstance(input_signal, np.ndarray):
        raise ValueError("Input signal must be a NumPy array")    

    standardized_signal = (input_signal - np.mean(input_signal)) / np.std(input_signal)

    return standardized_signal