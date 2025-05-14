import numpy as np
from Data.correct_img import My_img, NET


def fitness_function(weights):
    weights = np.array(weights, dtype=np.float32)
    correct_score = np.dot(weights, My_img)

    if correct_score < NET:
        return 0.0

    total_combinations = 2 ** 16
    correct_index = int(''.join(map(str, My_img)), 2)

    penalty_sum = 0.0
    penalty_count = 0

    for i in range(total_combinations):
        if i == correct_index:
            continue
        image_bits = np.array([int(b) for b in format(i, '016b')], dtype=np.uint8)
        score = np.dot(weights, image_bits)
        if score >= NET:
            penalty_sum += score - NET  # Насколько сильно оно ошибается
            penalty_count += 1

    if penalty_count == 0:
        return 1.0  # Идеальное решение

    penalty = penalty_sum / penalty_count
    fitness = 1.0 / (1.0 + penalty * penalty_count)  # Чем выше среднее превышение, тем хуже

    return round(fitness, 8)
