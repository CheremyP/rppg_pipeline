import numpy as np
from scipy import signal

#TODO: Add heartpy estimation aswell

def get_bpm(ppg_signal, fps= 30.0):
  sig = ppg_signal.copy()
  ppg_length = len(ppg_signal)
  win = signal.hann(ppg_signal.size)
  ppg_signal = ppg_signal - np.expand_dims(np.mean(ppg_signal, -1 ), -1 )
  ppg_signal = ppg_signal * win
  f, Pxx_den = signal.welch(ppg_signal, fps, nperseg=ppg_length)
  index = np.argmax(Pxx_den)
  HR_estimate =  round(f[index] * 60)
  return HR_estimate

def next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def calculate_fft_hr(ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    N = next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr

def calculate_peak_hr(ppg_signal, fs):
  """Calculate heart rate based on PPG using peak detection."""
  ppg_peaks, _ = signal.find_peaks(ppg_signal, height=0.2 ,distance=15)
  hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
  return hr_peak