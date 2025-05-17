import numpy as np
from Genetic.Methods.Fit_func import fitness_function_cached
import random
from Genetic.Methods.Utils import round_weights, zero_destroyer

def mutation_1(population, MUTATION_RATE, SHOW):
    """
    Сильная мутация - до ответа далеко
    Мутация в которой вес может поменяться от -1 до 1
    """
    new_chromosomes = []
    mutation_count = 0

    for chrome in population:
        if random.random() <= MUTATION_RATE:
            weights, fitness = chrome  # Распаковываем кортеж (weights, fitness)
            mutated_weights = weights.copy()  # Создаём копию, чтобы не менять исходный массив
            mutation_count += 1

            # Проходим по каждому весу в хромосоме
            for i in range(len(mutated_weights)):
                # Для каждого гена проверяем, нужно ли его мутировать
                if random.random() <= MUTATION_RATE:
                    mutated_weights[i] = np.random.uniform(-1.0, 1.0)  # Новый случайный вес

            mutated_weights = round_weights(mutated_weights)
            mutated_weights = zero_destroyer(mutated_weights)
            new_fitness = fitness_function_cached(mutated_weights)
            new_chromosomes.append((mutated_weights, new_fitness))

            if SHOW:
                # Вывод только если были мутации
                print(f"\nПроизошла мутация первого типа {mutation_count}:")
                print(
                    f"Было: {np.array2string(weights, formatter={'float_kind': lambda x: '%.8f' % x})} \n(Фитнес: {fitness:.8f})\n")
                print(
                    f"Стало: {np.array2string(mutated_weights, formatter={'float_kind': lambda x: '%.8f' % x})} \n(Фитнес: {new_fitness:.8f})\n")

    print(f"\nВсего мутаций первого типа для поколения: {mutation_count}")
    population.extend(new_chromosomes)
    return population

def mutation_2(population, MUTATION_RATE, SHOW):
    """
    Средняя/слабая мутация - когда близки к ответу
    Мы меняем значение на +-20% от его значения
    """
    new_chromosomes = []
    mutation_count = 0

    for chrome in population:
        if random.random() <= MUTATION_RATE:
            weights, fitness = chrome  # Распаковываем кортеж (weights, fitness)
            mutated_weights = weights.copy()  # Создаём копию, чтобы не менять исходный массив
            mutation_count += 1

            # Проходим по каждому весу в хромосоме
            for i in range(len(mutated_weights)):
                # Для каждого гена проверяем, нужно ли его мутировать
                if random.random() <= MUTATION_RATE:
                    if mutated_weights[i] > 0:
                        mutated_weights[i] = np.random.uniform(mutated_weights[i]*0.8, mutated_weights[i]*1.2)
                        if mutated_weights[i] > 1:
                            mutated_weights[i] = 1
                    elif mutated_weights[i] == 0:
                        mutated_weights[i] = np.random.uniform(-0.1, 0.1)
                    else:
                        mutated_weights[i] = np.random.uniform(mutated_weights[i] * 1.2, mutated_weights[i] * 0.8)
                        if mutated_weights[i] < -1:
                            mutated_weights[i] = -1

            mutated_weights = round_weights(mutated_weights)
            mutated_weights = zero_destroyer(mutated_weights)
            new_fitness = fitness_function_cached(mutated_weights)
            new_chromosomes.append((mutated_weights, new_fitness))

            if SHOW:
                # Вывод только если были мутации
                print(f"\nПроизошла мутация второго типа {mutation_count}:")
                print(
                    f"Было: {np.array2string(weights, formatter={'float_kind': lambda x: '%.8f' % x})} \n(Фитнес: {fitness:.8f})\n")
                print(
                    f"Стало: {np.array2string(mutated_weights, formatter={'float_kind': lambda x: '%.8f' % x})} \n(Фитнес: {new_fitness:.8f})\n")

    print(f"\nВсего мутаций второго типа для поколения: {mutation_count}")


    population.extend(new_chromosomes)
    return population


def mutation_3(population, MUTATION_RATE, SHOW):
    """
    Сильная мутация - далеко от ответа
    Может поменять знак у элемента веса
    """
    new_chromosomes = []
    mutation_count = 0

    for chrome in population:
        if random.random() <= MUTATION_RATE:
            weights, fitness = chrome  # Распаковываем кортеж (weights, fitness)
            mutated_weights = weights.copy()  # Создаём копию, чтобы не менять исходный массив
            mutation_count += 1

            # Проходим по каждому весу в хромосоме
            for i in range(len(mutated_weights)):
                # Для каждого гена проверяем, нужно ли его мутировать
                if random.random() <= MUTATION_RATE:
                    if mutated_weights[i] != 0:
                        mutated_weights[i] = mutated_weights[i] * -1  # Новый случайный вес
                    else:
                        mutated_weights[i] = np.random.uniform(-0.01, 0.01)

            mutated_weights = round_weights(mutated_weights)
            mutated_weights = zero_destroyer(mutated_weights)
            new_fitness = fitness_function_cached(mutated_weights)
            new_chromosomes.append((mutated_weights, new_fitness))

            if SHOW:
                # Вывод только если были мутации
                print(f"\nПроизошла мутация третьего типа {mutation_count}:")
                print(
                    f"Было: {np.array2string(weights, formatter={'float_kind': lambda x: '%.8f' % x})} \n(Фитнес: {fitness:.8f})\n")
                print(
                    f"Стало: {np.array2string(mutated_weights, formatter={'float_kind': lambda x: '%.8f' % x})} \n(Фитнес: {new_fitness:.8f})\n")

    print(f"\nВсего мутаций третьего типа для поколения: {mutation_count}")
    population.extend(new_chromosomes)
    return population
