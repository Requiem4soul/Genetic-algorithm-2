import numpy as np
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime
from Check_results import load_weights, evaluate_weights
from Genetic.Methods.Generate_population import generate_population
from Genetic.Methods.Mutation import mutation_1, mutation_2, mutation_3
from Genetic.Methods.Elitism import update_archive, elitism
from Genetic.Methods.Crossover import crossover_1, crossover_2, crossover_3
from Data.correct_img import My_img, NET

# Метка времени запуска
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Параметры
POPULATION_SIZE = 50
EPOCH = 10000
MUTATION_RATE = 0.4
SHOW = False
CHECK_RESULTS_ON = True
STATS_SHOW = True

# Инициализация популяции
population = generate_population(POPULATION_SIZE)

PAR = int(len(population)*0.25)

STATS = [] # Сохранение статистики обучения

start_time = time.time()
# Вывод начальной популяции
print("═" * 50)
print("Начальная популяция:\n")
for i, (weights, fitness) in enumerate(population):
    formatted_weights = [f"{w:.2f}" for w in weights]
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

    if STATS_SHOW:
        STATS.append((best_fitness, worst_fitness, avg_fitness)) # Сохраняем статистику

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
print(f"Обучение заняло {minutes} минут и {int(seconds)} секунд")

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

# Построение графика обучения
epochs = list(range(1, len(STATS) + 1))
best_vals = [stat[0] for stat in STATS]
worst_vals = [stat[1] for stat in STATS]
avg_vals = [stat[2] for stat in STATS]

plt.figure(figsize=(10, 6))
plt.plot(epochs, best_vals, label='Лучший фитнес', color='green')
plt.plot(epochs, worst_vals, label='Худший фитнес', color='red')
plt.plot(epochs, avg_vals, label='Средний фитнес', color='blue')
plt.xlabel('Эпоха')
plt.ylabel('Фитнес')
plt.title('Прогресс обучения по эпохам')
plt.legend()
plt.grid(True)
plt.tight_layout()
graph_path = f"Data/Statistic/Graphs/{timestamp}.png"
plt.savefig(graph_path)
if STATS_SHOW:
    plt.show()
print(f"График сохранён в '{graph_path}'")

# Сохранение метаданных обучения
metadata_path = f"Data/Statistic/Metadata/{timestamp}.txt"
with open(metadata_path, 'w', encoding='utf-8') as f:
    f.write("Информация об обучении\n\n")
    f.write(f"Правильная картинка: {My_img}\n")
    f.write(f"NET для активации: {NET}\n\n")
    f.write("═" * 40 + "\n")
    f.write(f"Популяция: {POPULATION_SIZE}\n")
    f.write(f"Максимум эпох: {EPOCH}\n")
    f.write(f"Итоговое количество эпох: {len(STATS)}\n")
    f.write(f"Коэффициент мутации: {MUTATION_RATE}\n")
    f.write(f"Время обучения: {int(minutes)} минут {int(seconds)} секунд\n\n")
    f.write("Лучший результат:\n")
    f.write(f"Фитнес: {best_chromosome[1]:.8f}\n")
    f.write("Веса:\n")
    for i, w in enumerate(best_chromosome[0]):
        f.write(f"  Ген {i + 1}: {w:.8f}\n")
print(f"Метаданные сохранены в '{metadata_path}'")