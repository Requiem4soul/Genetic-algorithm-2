def elitism(population, POPULATION_SIZE):
    """
    Оставляет только лучшие уникальные особи в размере POPULATION_SIZE
    """
    seen = set()
    unique_population = []

    for weights, fitness in population:
        key = tuple(weights.tolist())  # Преобразуем в хэшируемый вид
        if key not in seen:
            seen.add(key)
            unique_population.append((weights, fitness))

    # Сортируем по убыванию фитнеса
    unique_population.sort(key=lambda chromo: chromo[1], reverse=True)

    # Оставляем только нужное количество лучших
    new_population = unique_population[:POPULATION_SIZE]

    population.clear()
    population.extend(new_population)
