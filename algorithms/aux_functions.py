import numpy as np


def split_sequence(sequence, n_steps):
    sequence = np.asarray(sequence)
    n_windows = len(sequence) - n_steps
    windows = np.empty((n_windows, n_steps), dtype=sequence.dtype)
    windows[:, :] = sequence[:-n_steps][:, None]
    labels = sequence[n_steps:]
    return windows, labels

