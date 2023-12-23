import numpy as np
import scipy.signal as signal
from scipy import sparse

def detrend(input_signal, lambda_value = 100) -> np.array():
    if not isinstance(input_signal, np.ndarray):
        raise ValueError("Input signal must be a NumPy array")
    
    if input_signal.ndim != 2:
        raise ValueError("Input signal must be a 3D array with dimensions (RGB, frames)")
    
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal

def cheby2_bandpass_filter(input_signal, lowcut, highcut, fs, order=4):
    if not isinstance(input_signal, np.ndarray):
        raise ValueError("Input signal must be a NumPy array")
    
    if input_signal.ndim != 2:
        raise ValueError("Input signal must be a 3D array with dimensions (RGB, frames)")
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.cheby2(order, 5, [low, high], btype="band")
    y = signal.filtfilt(b, a, input_signal)
    return y

def hilbert_filter(input_signal, lowcut, highcut, fs, order=4):
    if not isinstance(input_signal, np.ndarray):
        raise ValueError("Input signal must be a NumPy array")
    
    if input_signal.ndim != 2:
        raise ValueError("Input signal must be a 3D array with dimensions (RGB, frames)")
    
    HilberFiltered = np.empty_like(input_signal, dtype=np.float64)

    for channel in range(3):
        analytic_signal = signal.hilbert(input_signal[:, channel]) 
        amplitude_envelope = np.abs(analytic_signal)
        HilberFiltered[:, channel] = input_signal[:, channel] / amplitude_envelope

    return HilberFiltered

def normalisation(input_signal):
    if not isinstance(input_signal, np.ndarray):
        raise ValueError("Input signal must be a NumPy array")
    
    if input_signal.ndim != 2:
        raise ValueError("Input signal must be a 3D array with dimensions (RGB, frames)")

    normalized_signal = (input_signal - np.min(input_signal, axis=-1, keepdims=True)) / (np.max(input_signal, axis=-1, keepdims=True) - 
                                                                                         np.min(input_signal, axis=-1, keepdims=True))
    return normalized_signal

def standardisation(input_signal):
    if not isinstance(input_signal, np.ndarray):
        raise ValueError("Input signal must be a NumPy array")

    if input_signal.ndim != 2:
        raise ValueError("Input signal must be a 3D array with dimensions (RGB, frames)")
    
    mean = np.mean(input_signal, axis=-1, keepdims=True)
    std = np.std(input_signal, axis=-1, keepdims=True)
    standardized_signal = (input_signal - mean) / std

    return standardized_signal