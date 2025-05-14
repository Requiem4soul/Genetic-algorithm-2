import numpy as np

MAX_ARCHIVE_SIZE = 1000

archive_dict = {}

def update_archive(population):
    """
    Добавляет уникальные хромосомы из популяции в архив.
    Если архив превышает MAX_ARCHIVE_SIZE, сохраняет только лучших.
    """
    for weights, fitness in population:
        key = tuple(weights.tolist())
        if key not in archive_dict or archive_dict[key] < fitness:
            archive_dict[key] = fitness

    # Если архив переполнен, оставляем только top-N по фитнесу
    if len(archive_dict) > MAX_ARCHIVE_SIZE:
        # Сортировка и обрезка
        top_items = sorted(archive_dict.items(), key=lambda x: x[1], reverse=True)[:MAX_ARCHIVE_SIZE]
        archive_dict.clear()
        archive_dict.update(top_items)

def elitism(_, POPULATION_SIZE, SHOW):
    """
    Возвращает POPULATION_SIZE лучших уникальных хромосом из архива.
    """
    sorted_chromosomes = sorted(archive_dict.items(), key=lambda x: x[1], reverse=True)
    selected = sorted_chromosomes[:POPULATION_SIZE]

    if SHOW:
        print("\nВывод хромосом после элитизма:")
        for i, (weights_tuple, fitness) in enumerate(selected, start=1):
            formatted_weights = [f"{w:.8f}" for w in weights_tuple]
            print(f"Хромосома {i}: [{', '.join(formatted_weights)}]")
            print(f"Фитнес: {fitness:.8f}\n")

    return [(np.array(weights), fitness) for weights, fitness in selected]
