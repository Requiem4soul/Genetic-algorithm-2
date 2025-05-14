import numpy as np
from Genetic.Methods.Fit_func import fitness_function_cached


def generate_population(population_size):
    """
    Генерирует начальную популяцию с вычисленными фитнесами.Кол-во задавать через main
    Возвращает: список кортежей [(weights, fitness), ...], где weights — numpy массив
                из 16 весов в диапазоне [-1, 1], fitness — float (результат fitness_function).
    """
    population = []
    for _ in range(population_size):
        # Создаём хромосому: 16 случайных весов в [-1, 1]
        weights = np.random.uniform(low=-0.5, high=0.5, size=16).astype(np.float32)
        # Вычисляем фитнес для этой хромосомы
        fitness = fitness_function_cached(weights)
        # Добавляем кортеж (weights, fitness) в популяцию
        population.append((weights, fitness))

    return population

