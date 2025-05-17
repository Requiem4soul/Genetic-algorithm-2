import numpy as np
from Genetic.Methods.Utils import My_img_mask
from Data.correct_img import MAX_SUM_NET

# Старая с пенальти. Не отражала реальных результатов. С пенальти показывает что уже около 1, а по факту лишь 16% верно распознаёт
# Но оказалосб что для лёгких и средних значений лучше её использовать
# Короче просто лучше не использовать сложные случаи искомого изображения

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

    # Мне надоело смотреть как он пытается вес 0.35 убрать уже 20 минут
    error_active_than_NET = 0
    active_sum = np.sum(weights[My_img_mask])
    if active_sum > MAX_SUM_NET:
        error_active_than_NET = active_sum - NET


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
            # Если в весах нули, но уже по идее это пофиксил. Но пусть лучше останется
            if penalty == 0.0:
                fitness = 1.0 / (1.0 + penalty_count + error_active_than_NET)
            else:
                fitness = 1.0 / (1.0 + penalty * penalty_count + error_active_than_NET)

        fitness = round(fitness, 8)

    # Сохраняем в кеш
    fitness_cache[key] = fitness
    return fitness


# import numpy as np
#
# # Кеш для уже вычисленных фитнесов
# fitness_cache = {}
#
# # Импорт изображения и порога
# from Data.correct_img import My_img, NET
#
# def fitness_function_cached(weights):
#     """
#     Строгая фитнес-функция:
#     - 0.0, если правильное изображение не проходит порог
#     - 1.0 / (количество ложных срабатываний + 1) в остальных случаях
#     """
#     key = tuple(np.round(weights, 5))  # округляем, чтобы убрать шум
#     if key in fitness_cache:
#         return fitness_cache[key]
#
#     weights = np.array(weights, dtype=np.float32)
#
#     # Проверка правильного изображения
#     if np.dot(weights, My_img) < NET:
#         fitness_cache[key] = 0.0
#         return 0.0
#
#     # Генерация всех бинарных изображений
#     correct_int = int(''.join(map(str, My_img)), 2)
#     false_positives = 0
#
#     for i in range(2 ** 16):
#         if i == correct_int:
#             continue
#         bits = np.array([int(b) for b in format(i, '016b')], dtype=np.uint8)
#         score = np.dot(weights, bits)
#         if score >= NET:
#             false_positives += 1
#
#     fitness = 1.0 / (false_positives + 1)
#     fitness_cache[key] = fitness
#     return fitness
#
