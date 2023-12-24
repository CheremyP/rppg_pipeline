from scipy import signal
import numpy as np
import heartpy as hp


def get_bpm(ppg_signal: np.array(), sampling_rate=30.0) -> float:
  """
  Estimate the heart rate from a PPG signal using Welch's method.
  
  Parameters:
  ppg_signal (np.array): The PPG signal  of shape (1, frames).
  fps (float): The sampling rate of the PPG signal.

  Returns:
  float: The estimated heart rate in beats per minute.
  """
  ppg_length = len(ppg_signal)
  window = signal.windows.hann(ppg_length)
  ppg_signal = ppg_signal - np.mean(ppg_signal)
  ppg_signal = ppg_signal * window
  frequencies, power_spectral_density = signal.welch(ppg_signal, sampling_rate, nperseg=ppg_length)
  max_power_index = np.argmax(power_spectral_density)
  heart_rate_estimate = round(frequencies[max_power_index] * 60)
  return heart_rate_estimate

def calculate_fft_hr(ppg_signal: np.array(), sampling_rate=30, low_pass=0.75, high_pass=2.5) -> float:
  """
  Calculate heart rate from a PPG signal using Fast Fourier Transform (FFT).
  
  Parameters:
  ppg_signal (np.array): The PPG signal of shape (1, frames).
  sampling_rate (float): The sampling rate of the PPG signal.
  low_pass (float): The lower bound of the frequency range of interest.
  high_pass (float): The upper bound of the frequency range of interest.

  Returns:
  float: The estimated heart rate in beats per minute.
  """
  fft_length = np.fft.next_fast_len(ppg_signal.shape[1])
  frequencies, power_spectral_density = signal.periodogram(ppg_signal, fs=sampling_rate, nfft=fft_length, detrend=False)
  frequency_mask = (frequencies >= low_pass) & (frequencies <= high_pass)
  masked_frequencies = frequencies[frequency_mask]
  masked_power_spectral_density = power_spectral_density[frequency_mask]
  fft_heart_rate = masked_frequencies[np.argmax(masked_power_spectral_density)] * 60
  return fft_heart_rate

def calculate_peak_hr(ppg_signal, sampling_rate=30) -> float:
  """
  Calculate heart rate from a PPG signal using peak detection.
  
  Parameters:
  ppg_signal (np.array): The PPG signal of shape (1, frames).
  sampling_rate (float): The sampling rate of the PPG signal.

  Returns:
  float: The estimated heart rate in beats per minute.
  """
  peaks, _ = signal.find_peaks(ppg_signal, height=0.2, distance=15)
  heart_rate_peak = 60 / (np.mean(np.diff(peaks)) / sampling_rate)
  return heart_rate_peak

def Calculate_heartpy_hr(ppg_signal, sampling_rate=30) -> float:
  """
  Calculate heart rate from a PPG signal using heartpy.
  
  Parameters:
  ppg_signal (np.array): The PPG signal of shape (1, frames).
  sampling_rate (float): The sampling rate of the PPG signal.

  Returns:
  float: The estimated heart rate in beats per minute.
  """
  _, measures = hp.process(ppg_signal, sampling_rate)
  heartpy_hr = measures['bpm']
  return heartpy_hr