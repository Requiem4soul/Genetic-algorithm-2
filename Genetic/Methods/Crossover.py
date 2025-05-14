import numpy as np
from Genetic.Methods.Fit_func import fitness_function

def crossover_1(population, SHOW):
    """
    Стандартный кроссовер с точностью до двух знаков после запятой.
    """
    sorted_pop = sorted(population, key=lambda x: x[1])
    new_offspring = []
    pop_half = len(population) // 2
    crossover_count = 0
    len_popul = len(population)

    for i in range(0, pop_half*2, 2):
        parent1, parent2 = sorted_pop[i], sorted_pop[i + 1]
        weights1, fit1 = parent1
        weights2, fit2 = parent2

        child1_weights = np.zeros_like(weights1)
        child2_weights = np.zeros_like(weights2)

        if SHOW:
            # Вывод информации о родителях
            print(f"\nКроссовер {crossover_count + 1}:")
            print(
                f"Родитель 1: {np.array2string(weights1, formatter={'float_kind': lambda x: '%.8f' % x})} \nФитнес: {fit1:.8f}")
            print(
                f"Родитель 2: {np.array2string(weights2, formatter={'float_kind': lambda x: '%.8f' % x})} \nФитнес: {fit2:.8f}")

        for gene_idx in range(16):
            # Обработка отрицательных значений
            sign1 = -1 if weights1[gene_idx] < 0 else 1
            sign2 = -1 if weights2[gene_idx] < 0 else 1

            # Работаем с модулями значений
            val1 = abs(round(weights1[gene_idx], 2))
            val2 = abs(round(weights2[gene_idx], 2))

            # Конвертация в бинарный формат (7 бит)
            gene1 = int(val1 * 100)
            gene2 = int(val2 * 100)

            bin1 = format(gene1, '07b')
            bin2 = format(gene2, '07b')

            # Кроссовер (4|3 бита)
            new_bin1 = bin1[:4] + bin2[4:]
            new_bin2 = bin2[:4] + bin1[4:]

            # Обратная конвертация с восстановлением знака
            new_val1 = int(new_bin1, 2) / 100 * sign1
            new_val2 = int(new_bin2, 2) / 100 * sign2

            child1_weights[gene_idx] = new_val1
            child2_weights[gene_idx] = new_val2

        # Расчет фитнеса
        child1_fit = fitness_function(child1_weights)
        child2_fit = fitness_function(child2_weights)

        new_offspring.extend([
            (child1_weights, child1_fit),
            (child2_weights, child2_fit)
        ])

        if SHOW:
            # Вывод информации о потомках
            print(
                f"Потомок 1: {np.array2string(child1_weights, formatter={'float_kind': lambda x: '%.8f' % x})} \nФитнес: {child1_fit:.8f}")
            print(
                f"Потомок 2: {np.array2string(child2_weights, formatter={'float_kind': lambda x: '%.8f' % x})} \nФитнес: {child2_fit:.8f}")

        crossover_count += 1

    population.extend(new_offspring)
    print(f"\nВсего выполнено кроссоверов: {crossover_count}")
    print(f"Хромосом в популяции: {len_popul}\n")
    return population


def crossover_2(population, SHOW):
    """
    Кроссовер с точностью до одного знака после запятой (3 бита).
    """
    sorted_pop = sorted(population, key=lambda x: x[1])
    new_offspring = []
    pop_half = len(population) // 2
    crossover_count = 0
    len_popul = len(population)

    for i in range(0, pop_half*2, 2):
        parent1, parent2 = sorted_pop[i], sorted_pop[i + 1]
        weights1, fit1 = parent1
        weights2, fit2 = parent2

        child1_weights = np.zeros_like(weights1)
        child2_weights = np.zeros_like(weights2)

        if SHOW:
            # Вывод информации о родителях
            print(f"\nКроссовер {crossover_count + 1}:")
            print(
                f"Родитель 1: {np.array2string(weights1, formatter={'float_kind': lambda x: '%.8f' % x})} \nФитнес: {fit1:.8f}")
            print(
                f"Родитель 2: {np.array2string(weights2, formatter={'float_kind': lambda x: '%.8f' % x})} \nФитнес: {fit2:.8f}")

        for gene_idx in range(16):
            # Обработка отрицательных значений
            sign1 = -1 if weights1[gene_idx] < 0 else 1
            sign2 = -1 if weights2[gene_idx] < 0 else 1

            # Работаем с модулями значений, но только с одним знаком после запятой
            val1 = abs(round(weights1[gene_idx], 1))
            val2 = abs(round(weights2[gene_idx], 1))

            # Конвертация в бинарный формат (3 бита)
            gene1 = int(val1 * 10)
            gene2 = int(val2 * 10)

            bin1 = format(gene1, '03b')
            bin2 = format(gene2, '03b')

            # Кроссовер (2|1 бита)
            new_bin1 = bin1[:2] + bin2[2:]
            new_bin2 = bin2[:2] + bin1[2:]

            # Обратная конвертация с восстановлением знака
            new_val1 = int(new_bin1, 2) / 10 * sign1
            new_val2 = int(new_bin2, 2) / 10 * sign2

            child1_weights[gene_idx] = new_val1
            child2_weights[gene_idx] = new_val2

        # Расчет фитнеса
        child1_fit = fitness_function(child1_weights)
        child2_fit = fitness_function(child2_weights)

        new_offspring.extend([
            (child1_weights, child1_fit),
            (child2_weights, child2_fit)
        ])

        if SHOW:
            # Вывод информации о потомках
            print(
                f"Потомок 1: {np.array2string(child1_weights, formatter={'float_kind': lambda x: '%.8f' % x})} \nФитнес: {child1_fit:.8f}")
            print(
                f"Потомок 2: {np.array2string(child2_weights, formatter={'float_kind': lambda x: '%.8f' % x})} \nФитнес: {child2_fit:.8f}")

        crossover_count += 1

    population.extend(new_offspring)
    print(f"\nВсего выполнено кроссоверов: {crossover_count}")
    print(f"Хромосом в популяции: {len_popul}\n")
    return population


def crossover_3(population, SHOW):
    """
    Кроссовер с обменом весов (деление пополам и обмен частей).
    """
    sorted_pop = sorted(population, key=lambda x: x[1])
    new_offspring = []
    pop_half = len(population) // 2
    crossover_count = 0
    len_popul = len(population)

    for i in range(0, pop_half*2, 2):
        parent1, parent2 = sorted_pop[i], sorted_pop[i + 1]
        weights1, fit1 = parent1
        weights2, fit2 = parent2

        child1_weights = np.zeros_like(weights1)
        child2_weights = np.zeros_like(weights2)

        if SHOW:
            # Вывод информации о родителях
            print(f"\nКроссовер {crossover_count + 1}:")
            print(
                f"Родитель 1: {np.array2string(weights1, formatter={'float_kind': lambda x: '%.8f' % x})} \nФитнес: {fit1:.8f}")
            print(
                f"Родитель 2: {np.array2string(weights2, formatter={'float_kind': lambda x: '%.8f' % x})} \nФитнес: {fit2:.8f}")

        # Разделяем веса пополам и меняем их
        child1_weights[:8] = weights1[:8]
        child1_weights[8:] = weights2[8:]
        child2_weights[:8] = weights2[:8]
        child2_weights[8:] = weights1[8:]

        # Расчет фитнеса
        child1_fit = fitness_function(child1_weights)
        child2_fit = fitness_function(child2_weights)

        new_offspring.extend([
            (child1_weights, child1_fit),
            (child2_weights, child2_fit)
        ])

        if SHOW:
            # Вывод информации о потомках
            print(
                f"Потомок 1: {np.array2string(child1_weights, formatter={'float_kind': lambda x: '%.8f' % x})} \nФитнес: {child1_fit:.8f}")
            print(
                f"Потомок 2: {np.array2string(child2_weights, formatter={'float_kind': lambda x: '%.8f' % x})} \nФитнес: {child2_fit:.8f}")

        crossover_count += 1

    population.extend(new_offspring)
    print(f"\nВсего выполнено кроссоверов: {crossover_count}")
    print(f"Хромосом в популяции: {len_popul}\n")
    return population
