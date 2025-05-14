import numpy as np
from Genetic.Methods.Fit_func import fitness_function
from Genetic.Methods.Generate_population import generate_population
from Genetic.Methods.Mutation import mutation
from Genetic.Methods.Elitism import elitism
from Genetic.Methods.Crossover import crossover_1, crossover_2, crossover_3
from Data.correct_img import My_img, NET
from tqdm import tqdm

# Параметры
POPULATION_SIZE = 15
EPOCH = 100
MUTATION_RATE = 0.3
SHOW = False

# Инициализация популяции
population = generate_population(POPULATION_SIZE)

# Вывод начальной популяции
print("═" * 50)
print("Начальная популяция:\n")
for i, (weights, fitness) in enumerate(population):
    formatted_weights = [f"{w:.8f}" for w in weights]
    print(f"Хромосома {i + 1}: [{', '.join(formatted_weights)}]")
    print(f"Фитнес: {fitness:.8f}\n")

# Основной цикл обучения
for epoch in range(EPOCH):
    print("═" * 50)
    print(f"Эпоха {epoch + 1}/{EPOCH}")

    population = mutation(population, MUTATION_RATE, SHOW)
    crossover_3(population, SHOW)
    elitism(population, POPULATION_SIZE)

    # Статистика по эпохе
    fitness_values = [chromo[1] for chromo in population]
    best_fitness = max(fitness_values)
    worst_fitness = min(fitness_values)
    avg_fitness = sum(fitness_values) / len(fitness_values)

    best_index = fitness_values.index(best_fitness)
    worst_index = fitness_values.index(worst_fitness)

    print(f"\nСтатистика эпохи {epoch + 1}:")
    print(f"Лучший:  {best_fitness:.8f} (Хромосома {best_index + 1})")
    print(f"Худший:  {worst_fitness:.8f} (Хромосома {worst_index + 1})")
    print(f"Средний: {avg_fitness:.8f}\n")

# Итоговый результат
print("═" * 50)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО\n")

best_chromosome = max(population, key=lambda x: x[1])
print("Лучший результат обучения:")
print(f"Фитнес: {best_chromosome[1]:.8f}")
print("Веса:")
for i, w in enumerate(best_chromosome[0]):
    print(f"Ген {i + 1}: {w:.8f}")

# Сохраняем только веса
best_weights = best_chromosome[0]
np.savetxt('best_weights.txt', best_weights, fmt='%.8f')
print("\nВеса лучшей хромосомы сохранены в 'best_weights.txt'")