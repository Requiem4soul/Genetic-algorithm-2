import numpy as np

def round_weights(weights, decimals=2):
    if isinstance(weights, np.ndarray):
        return np.round(weights, decimals)
    return [round(w, decimals) for w in weights]