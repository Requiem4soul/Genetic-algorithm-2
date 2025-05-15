import numpy as np

# Кеш для уже вычисленных фитнесов
fitness_cache = {}

# Импорт изображения и порога
from Data.correct_img import My_img, NET

def fitness_function_cached(weights):
    """
    Фитнес функция, которая оценивает насколько близко к NET значение активации (для правильного и ложных изображений)
    Если правильное вообще не определилось - сразу 0
    """
    key = tuple(np.round(weights, 5))  # округляем, чтобы убрать шум
    if key in fitness_cache:
        return fitness_cache[key]

    weights = np.array(weights, dtype=np.float32)

    # Проверка правильного изображения
    net_correct = np.dot(weights, My_img)
    if net_correct < NET:
        fitness = 0.0
    else:
        # Генерация всех бинарных изображений
        total_combinations = 2 ** 16
        correct_int = int(''.join(map(str, My_img)), 2)

        penalty_sum = 0.0
        penalty_count = 0

        for i in range(total_combinations):
            if i == correct_int:
                continue
            bits = np.array([int(x) for x in format(i, '016b')], dtype=np.uint8)
            score = np.dot(weights, bits)
            if score >= NET:
                penalty_sum += score - NET  # насколько сильно превысили порог
                penalty_count += 1

        if penalty_count == 0:
            fitness = 1.0
        else:
            penalty = penalty_sum / penalty_count
            if penalty == 0.0:
                fitness = 1.0 / (1.0 + penalty_count)
            else:
                fitness = 1.0 / (1.0 + penalty * penalty_count)

        fitness = round(fitness, 8)

    # Сохраняем в кеш
    fitness_cache[key] = fitness
    return fitness
