import numpy as np
import scipy.signal as signal
import math 

def calculate_bvp_chrome(rgb_signal: np.array, low_pass_filter = 0.7, high_pass_filter = 2.5, sampling_frequency = 30) -> np.array:
    """ Calculate the blood volume pulse (BVP) signal from a RGB signal using the CHROME method.

    Parameters:
        rgb_signal (np.array): The RGB signal of shape (frames, 3).
        low_pass_filter (float): The low pass filter frequency.
        high_pass_filter (float): The high pass filter frequency.
        sampling_frequency (int): The sampling frequency.

    Returns:
        np.array: The estimated BVP signal of shape (1, frames).    
    """

    window_seconds = 1.6
    num_frames = rgb_signal.shape[0]
    nyquist_frequency = 0.5 * sampling_frequency
    cheby2_bandpass_filter = signal.cheby2(3, 5, [low_pass_filter/nyquist_frequency, high_pass_filter/nyquist_frequency], btype="band")

    window_length = math.ceil(window_seconds * sampling_frequency)
    if(window_length % 2):
        window_length = window_length + 1
    num_windows = math.floor((num_frames - window_length // 2) / (window_length // 2))
    total_length = (window_length // 2) * (num_windows + 1)
    bvp_signal = np.zeros(total_length)

    window_start = 0
    window_middle = int(window_start + window_length // 2)
    window_end = window_start + window_length

    for i in range(num_windows):
        rgb_base = np.mean(rgb_signal[window_start:window_end, :], axis=0)
        rgb_normalized = np.zeros((window_end - window_start, 3))
        for frame_index in range(window_start, window_end):
            rgb_normalized[frame_index - window_start] = np.true_divide(rgb_signal[frame_index], rgb_base) - 1
        x_signal = np.squeeze(3 * rgb_normalized[:, 0] - 2 * rgb_normalized[:, 1])
        y_signal = np.squeeze(1.5 * rgb_normalized[:, 0] + rgb_normalized[:, 1] - 1.5 * rgb_normalized[:, 2])
        x_filtered = signal.filtfilt(cheby2_bandpass_filter[0], cheby2_bandpass_filter[1], x_signal, axis=0)
        y_filtered = signal.filtfilt(cheby2_bandpass_filter[0], cheby2_bandpass_filter[1], y_signal)

        alpha = np.std(x_filtered) / np.std(y_filtered)
        window_signal = x_filtered - alpha * y_filtered
        window_signal = np.multiply(window_signal, signal.windows.hann(window_length))

        if(i == -1):
            bvp_signal = window_signal
        else:
            temp = window_signal[:int(window_length // 2)]
            bvp_signal[window_start:window_middle] = bvp_signal[window_start:window_middle] + window_signal[:int(window_length // 2)]
            bvp_signal[window_middle:window_end] = window_signal[int(window_length // 2):]
        window_start = window_middle
        window_middle = window_start + window_length // 2
        window_end = window_start + window_length

    return bvp_signal