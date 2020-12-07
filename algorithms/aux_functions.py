import numpy as np

# Adapted from the machinelearningmastery website:
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
def split_sequence(sequence, n_steps, n_intervals=1):
    X = list()
    for i in np.arange(0, len(sequence), n_intervals):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        X.append(seq_x)
    return np.array(X)