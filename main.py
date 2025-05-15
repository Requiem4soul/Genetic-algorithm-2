import numpy as np
import time
import os
from Check_results import load_weights, evaluate_weights
from Genetic.Methods.Generate_population import generate_population
from Genetic.Methods.Mutation import mutation_1, mutation_2, mutation_3
from Genetic.Methods.Elitism import update_archive, elitism
from Genetic.Methods.Crossover import crossover_1, crossover_2, crossover_3


# Параметры
POPULATION_SIZE = 50
EPOCH = 500
MUTATION_RATE = 0.3
SHOW = False
CHECK_RESULTS_ON = True

# Инициализация популяции
population = generate_population(POPULATION_SIZE)

PAR = int(len(population)*0.2)
start_time = time.time()
# Вывод начальной популяции
print("═" * 50)
print("Начальная популяция:\n")
for i, (weights, fitness) in enumerate(population):
    formatted_weights = [f"{w:.8f}" for w in weights]
    print(f"Хромосома {i + 1}: [{', '.join(formatted_weights)}]")
    print(f"Фитнес: {fitness:.8f}\n")

# Основной цикл обучения
for epoch in range(EPOCH):
    epoch_start_time = time.time()

    print("═" * 50)
    print(f"Эпоха {epoch + 1}/{EPOCH}")

    population = mutation_1(population, MUTATION_RATE, SHOW)
    population = mutation_2(population, MUTATION_RATE, SHOW)
    population = mutation_3(population, MUTATION_RATE, SHOW)
    update_archive(population)
    population = crossover_3(population, SHOW, PAR)
    update_archive(population)
    population = crossover_2(population, SHOW, PAR)
    update_archive(population)
    population = crossover_1(population, SHOW, PAR)
    update_archive(population)
    population = elitism(population, POPULATION_SIZE, SHOW)

    # Статистика по эпохе
    fitness_values = [chromo[1] for chromo in population]
    best_fitness = max(fitness_values)
    worst_fitness = min(fitness_values)
    avg_fitness = sum(fitness_values) / len(fitness_values)

    best_index = fitness_values.index(best_fitness)
    worst_index = fitness_values.index(worst_fitness)

    best_chromosome = population[best_index]

    print(f"\nСтатистика эпохи {epoch + 1}:")
    print(f"Лучший:  {best_fitness:.8f} (Хромосома {best_index + 1})")
    print(f"Худший:  {worst_fitness:.8f} (Хромосома {worst_index + 1})")
    print(f"Средний: {avg_fitness:.8f}\n")

    epoch_duration = time.time() - epoch_start_time
    print(f"Время выполнения эпохи: {epoch_duration:.3f} секунд\n")

    # Найдено лучшее решение
    if best_fitness == 1.0:
        print("\n Найдено идеальное решение! Обучение завершается досрочно.\n")
        np.savetxt('Data/Save_weights/best_before_end_weights.txt', best_chromosome[0], fmt='%.8f')
        break

    # Сохраняем лучшие веса каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        np.savetxt('Data/Save_weights/checkpoint_weights.txt', best_chromosome[0], fmt='%.8f')
        print(f"Контрольная точка: временное сохранение 'checkpoint_weights.txt'")

# Итоговый результат
print("═" * 50)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО\n")

end_time = time.time() - start_time
minutes = end_time // 60
seconds = end_time % 60
print(f"Обучение заняло {minutes} минут и {seconds} секунд")

best_chromosome = max(population, key=lambda x: x[1])
print("Лучший результат обучения:")
print(f"Фитнес: {best_chromosome[1]:.8f}")
print("Веса:")
for i, w in enumerate(best_chromosome[0]):
    print(f"Ген {i + 1}: {w:.8f}")

# Сохраняем только веса
best_weights = best_chromosome[0]
np.savetxt('Data/Save_weights/best_weights.txt', best_weights, fmt='%.8f')
print("\nВеса лучшей хромосомы сохранены в 'best_weights.txt'")

if CHECK_RESULTS_ON:
    # Проверка финального решения
    print("\nПроверка финального решения")

    # Определяем, какой файл использовать
    if os.path.exists("Data/Save_weights/best_weights.txt"):
        weights = load_weights("Data/Save_weights/best_weights.txt")
        print("Используются веса из 'best_weights.txt'")
    elif os.path.exists("Data/Save_weights/best_before_end_weights.txt"):
        weights = load_weights("Data/Save_weights/best_before_end_weights.txt")
        print("Используются веса из 'best_before_end_weights.txt'")
    else:
        print("Не найден файл с весами для проверки.")
        weights = None

    # Выполнение оценки, если веса загружены
    if weights is not None:
        evaluate_weights(weights)