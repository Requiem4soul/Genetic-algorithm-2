import numpy as np
from Genetic.Methods.Fit_func import fitness_function
import random

def mutation(population, MUTATION_RATE, SHOW):
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

            new_fitness = fitness_function(mutated_weights)
            new_chromosomes.append((mutated_weights, new_fitness))

            if SHOW:
                # Вывод только если были мутации
                print(f"\nПроизошла мутация {mutation_count}:")
                print(
                    f"Было: {np.array2string(weights, formatter={'float_kind': lambda x: '%.8f' % x})} \n(Фитнес: {fitness:.8f})\n")
                print(
                    f"Стало: {np.array2string(mutated_weights, formatter={'float_kind': lambda x: '%.8f' % x})} \n(Фитнес: {new_fitness:.8f})\n")

    print(f"\nВсего мутаций для поколения: {mutation_count}")


    population.extend(new_chromosomes)
    return population


