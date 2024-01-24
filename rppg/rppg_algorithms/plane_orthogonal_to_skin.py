import numpy as np
import math

def plane_orthoganal_to_skin(rgb_signal):
    """ Calculate the blood volume pulse (BVP) signal from a RGB signal using the plane orthogonal to skin method.

    Parameters:
        rgb_signal (np.array): The RGB signal of shape (frames, 3).

    Returns:
        np.array: The estimated BVP signal of shape (1, frames).    
    """
    frame_rate = 30
    window_seconds = 1.6
    num_samples = rgb_signal.shape[0]
    bvp_signal = np.zeros((1, num_samples))
    window_size = math.ceil(window_seconds * frame_rate)

    for sample_index in range(num_samples):
        window_start = sample_index - window_size
        if window_start >= 0:
            normalized_rgb = np.true_divide(rgb_signal[window_start:sample_index, :], np.mean(rgb_signal[window_start:sample_index, :], axis=0))
            normalized_rgb = np.mat(normalized_rgb).H
            skin_projection = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), normalized_rgb)
            skin_projection_adjusted = skin_projection[0, :] + (np.std(skin_projection[0, :]) / np.std(skin_projection[1, :])) * skin_projection[1, :]
            mean_adjusted_projection = np.mean(skin_projection_adjusted)
            for temp_index in range(skin_projection_adjusted.shape[1]):
                skin_projection_adjusted[0, temp_index] = skin_projection_adjusted[0, temp_index] - mean_adjusted_projection
            bvp_signal[0, window_start:sample_index] = bvp_signal[0, window_start:sample_index] + (skin_projection_adjusted[0])

    return bvp_signal