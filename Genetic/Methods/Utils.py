import numpy as np
from Data.correct_img import My_img


def round_weights(weights, decimals=2):
    if isinstance(weights, np.ndarray):
        return np.round(weights, decimals)
    return [round(w, decimals) for w in weights]

def zero_destroyer(weights):
    """
    Заменяет нули в весах на случайные значения в диапазоне (-0.01, 0.01).
    """
    zeros_mask = weights == 0
    weights[zeros_mask] = np.random.uniform(-0.01, 0.01, size=np.sum(zeros_mask))
    return weights

My_img_mask = (My_img == 1)
