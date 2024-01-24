import warnings
from scipy import signal
import numpy as np
import heartpy as hp

def get_bpm(ppg_signal, sampling_rate=30.0) -> float:
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


def next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def calculate_fft_hr(ppg_signal, sampling_rate=30, low_pass=0.75, high_pass=2.5):
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

  # Ignore UserWarning
  warnings.filterwarnings("ignore", category=UserWarning)

  N = next_power_of_2(ppg_signal.shape[0])

  f_ppg, pxx_ppg = signal.periodogram(ppg_signal, fs=sampling_rate, nfft=N, detrend=False)
  fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
  mask_ppg = np.take(f_ppg, fmask_ppg)
  mask_pxx = np.take(pxx_ppg, fmask_ppg)

  fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
  # Reset warnings to default behavior if needed
  warnings.resetwarnings()

  return fft_hr

def calculate_peak_hr(ppg_signal, sampling_rate=30) -> float:
  """
  Calculate heart rate from a PPG signal using peak detection.
  
  Parameters:
    ppg_signal (np.array): The PPG signal of shape (1, frames).
    sampling_rate (float): The sampling rate of the PPG signal.

  Returns:
    float: The estimated heart rate in beats per minute.
  """


  # Ignore ComplexWarning
  warnings.filterwarnings("ignore")

  peaks, _ = signal.find_peaks(ppg_signal, height=0.2, distance=15)
  heart_rate_peak = 60 / (np.mean(np.diff(peaks)) / sampling_rate)

  # Reset warnings to default behavior if needed
  warnings.resetwarnings()

  return heart_rate_peak

def calculate_heartpy_hr(ppg_signal, sampling_rate=30) -> float:
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

def calculate_hr(ppg_signal, sampling_rate=30) -> float:
  """ c  def calculate_hr_autocorrelation(ppg_signal, sampling_rate=30) -> float:
    Calculate heart rate from a PPG signal using autocorrelation.
    
    Parameters:
      ppg_signal (np.array): The PPG signal of shape (1, frames).
      sampling_rate (float): The sampling rate of the PPG signal.

    Returns:
      float: The estimated heart rate in beats per minute.
    """
  
  autocorr = np.correlate(ppg_signal, ppg_signal, mode='full')
  autocorr = autocorr[len(autocorr)//2:]
  peaks, _ = signal.find_peaks(autocorr)
  heart_rate_autocorr = 60 / (np.mean(np.diff(peaks)) / sampling_rate)
  return heart_rate_autocorr
