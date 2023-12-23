#TODO: Averaging/Standardisation
def detrend(input_signal, lambda_value = 100):
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
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.cheby2(order, 5, [low, high], btype="band")
    y = signal.filtfilt(b, a, input_signal)
    return y

def hilbert_filter(input_signal, lowcut, highcut, fs, order=4):
    if input_signal != 3:
        return ValueError
    
    for channel in range(3):
        analytic_signal = signal.hilbert(input_signal[:, channel])  # Apply hilbert to specific channel
        amplitude_envelope = np.abs(analytic_signal)  # Derive envelope signal
        HilberFiltered[:, channel] = BGRFiltered[:, channel] / amplitude_envelope
    
    return HilberFiltered

def normalisation(input_signal):
    return NotImplementedError

def standardisation(input_signal):
    standardised_signal = input_signal
    return standardised_signal