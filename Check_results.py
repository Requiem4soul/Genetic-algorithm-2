import numpy as np
from PIL import Image
import os
from Data.correct_img import NET, My_img

def load_weights(filename):
    return np.loadtxt(filename, dtype=np.float32)

def add_My_img(My_img):
    img_array = My_img.reshape((4, 4)) * 255
    save_path = "Data/Pictures_for_recognize/My_img.png"
    Image.fromarray(img_array.astype(np.uint8), mode='L').save(save_path)
    print(f"Изображение перезаписано")

def format_image(bits):
    # Представление 4x4
    return "\n".join(
        "".join(str(bits[i * 4 + j]) for j in range(4))
        for i in range(4)
    )

def evaluate_weights(weights):
    correct_output = np.dot(weights, My_img)
    if correct_output < NET:
        print(f"Ошибка: правильное изображение НЕ проходит порог. Выход = {correct_output:.4f}")
        return 0.0

    error_count = 0
    error_examples = []

    correct_int = int(''.join(map(str, My_img)), 2)

    for i in range(2 ** 16):
        if i == correct_int:
            continue
        bits = np.array([int(b) for b in format(i, '016b')], dtype=np.uint8)
        output = np.dot(weights, bits)
        if output >= NET:
            error_count += 1
            if len(error_examples) < 3:
                error_examples.append(bits.copy())

    fitness = 1.0 / (error_count + 1)
    print(f"Фитнес: {fitness:.8f}")
    print(f"Ложных срабатываний: {error_count}")

    if error_examples:
        print("\nПримеры ложных срабатываний:") # Надо проверять что вообще будет 3 неправильных изображения, но сейчас веса находят идеал
        for idx, example in enumerate(error_examples, 1):
            print(f"\nОшибка {idx}:")
            print(format_image(example))

    return fitness

# Пример вызова
if __name__ == "__main__":
    add_My_img(My_img)
    weights = load_weights("Data/Save_weights/best_weights.txt")
    evaluate_weights(weights)
