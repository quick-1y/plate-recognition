from ultralytics import YOLO
import os

if __name__ == '__main__':
    # Проверка текущей директории
    print(f"Current working directory: {os.getcwd()}")

    # Установка рабочей директории, если необходимо
    os.chdir('/')

    # Путь к предобученной модели для детекции типа транспортного средства
    pretrained_model_path = '../models/best.pt'

    # Загрузка предобученной модели с указанием количества классов
    model = YOLO(pretrained_model_path)

    # Определение относительного пути к конфигурационному файлу
    config_path = 'configs/plates_config.yaml'

    # Дообучение модели на объединенных данных
    results = model.train(
        data=config_path,
        epochs=5,
        imgsz=640,
        batch=16,
        name='plates',
        multi_scale=True  # Включение многомасштабного обучения
    )

    # Вывод результатов
    print(results)