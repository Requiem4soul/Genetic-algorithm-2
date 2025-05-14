import numpy as np
from Data.correct_img import My_img, NET


def fitness_function(weights):
    """
    Вычисляет фитнес для хромосомы (весов).
    Принимает вес (хромосому)
    Возвращает: float от 0 до 1, где 1 - только правильное изображение определено корректно.
    """
    # Преобразуем веса в numpy массив для удобства вычислений
    weights = np.array(weights, dtype=np.float32)

    # Проверяем правильное изображение
    NET_correct = np.sum(weights * My_img)
    if NET_correct < NET:
        return 0.0  # Правильное изображение не прошло порог, фитнес = 0

    # Считаем ошибки: сколько неправильных изображений имеют NET >= 0.4
    errors = 0
    total_combinations = 2 ** 16  # 65,536
    correct_binary = int(''.join(map(str, My_img)), 2)  # Число, соответствующее My_img

    for i in range(total_combinations):
        if i == correct_binary:
            continue  # Пропускаем правильное изображение
        # Преобразуем число в битовую маску
        binary = format(i, '016b')
        image = np.array([int(bit) for bit in binary], dtype=np.uint8)
        NET_value = np.sum(weights * image)
        if NET_value >= NET:
            errors += 1

    # Фитнес: 1 / (количество ошибок + 1)
    fitness = 1.0 / (errors + 1)
    fitness = round(fitness, 8)
    return fitness


# Тест: веса, где NET = 0.4 для правильного изображения
# weights = np.array([0.25] * 16, dtype=np.float32)
# fitness = fitness_function(weights)
# print(f"Фитнес: {fitness:.8f}")