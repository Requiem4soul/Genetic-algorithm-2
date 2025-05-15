import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Data.correct_img import NET

# Пути
WEIGHTS_DIR = "Data/Save_weights"
IMAGES_DIR = "Data/Pictures_for_recognize"

# Загрузка лучших весов
def load_best_weights():
    filenames = ["best_before_end_weights.txt", "best_weights.txt", "checkpoint_weights.txt"]
    for name in filenames:
        full_path = os.path.join(WEIGHTS_DIR, name)
        if os.path.exists(full_path):
            print(f"Используем веса: {name}")
            return np.loadtxt(full_path, dtype=np.float32), name
    print("Нет подходящих весов в папке.")
    return None, None

# Преобразование изображения в бинарный вектор
def image_to_vector(image_path):
    img = Image.open(image_path).convert("L").resize((4, 4))
    data = np.asarray(img)
    vector = (data < 128).astype(np.uint8).flatten()
    return vector

# Основная логика
def process_images_with_arrows():
    weights, _ = load_best_weights()
    if weights is None:
        return

    # Предобработка: собираем изображения и мета-данные
    image_infos = []
    for file in sorted(os.listdir(IMAGES_DIR)):
        if file.lower().endswith(".png"):
            path = os.path.join(IMAGES_DIR, file)
            vec = image_to_vector(path)
            activation = np.dot(weights, vec)
            is_correct = activation >= NET
            image_infos.append({
                "path": path,
                "activation": activation,
                "is_correct": is_correct,
            })

    if not image_infos:
        print("Нет изображений в папке.")
        return

    index = [0]  # оборачиваем в список для доступа внутри вложенной функции

    def update_display():
        info = image_infos[index[0]]
        img = Image.open(info["path"]).convert("L").resize((100, 100), Image.NEAREST)
        plt.clf()
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.title(
            f"{os.path.basename(info['path'])}\n"
            f"NET: {NET:.4f}, Актив: {info['activation']:.4f}\n"
            f"{'Совпадение' if info['is_correct'] else 'Не совпадает'}"
        )
        plt.axis("off")
        plt.subplots_adjust(top=0.8)
        plt.draw()

    def on_key(event):
        if event.key == "right" and index[0] < len(image_infos) - 1:
            index[0] += 1
            update_display()
        elif event.key == "left" and index[0] > 0:
            index[0] -= 1
            update_display()

    # Первая отрисовка и подключение событий
    fig = plt.figure()
    update_display()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    # После закрытия окна — финальная статистика:
    total = len(image_infos)
    correct = sum(1 for info in image_infos if info["is_correct"])
    print("═" * 40)
    print(f"Всего изображений: {total}")
    print(f"Угадано корректно: {correct}")
    print(f"Ошибок: {total - correct}")

    if correct == 1:
        print("Веса подходят — активировались один раз и правильно")
    elif correct > 1:
        print("Веса активируют несколько изображений — не подходят.")
    else:
        print("Нет ни одного правильного активационного ответа.")

# Запуск
process_images_with_arrows()
