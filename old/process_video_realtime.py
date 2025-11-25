import cv2
import os
import torch
from ultralytics import YOLO

# Путь к дообученной модели для детекции типа транспортного средства и номеров
#pretrained_model_path = '/runs/detect/plate/weights/best.pt'
pretrained_model_path = '../models/best.pt'

# Загрузка дообученной модели
model = YOLO(pretrained_model_path)

# Словарь с цветами для каждого класса
class_colors = {
    'car': (0, 255, 0),      # Зеленый
    'truck': (255, 0, 0),    # Синий
    'van': (0, 0, 255),      # Красный
    'bus': (255, 255, 0),    # Желтый
    'motorbike': (255, 0, 255),    # Фиолетовый
    'threewheel': (0, 255, 255),   # Голубой
    'plate': (255, 255, 255),    # Белый
}

# Функция для обработки видео в реальном времени
def process_video_realtime(video_path):
    # Открытие видеофайла
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Получение начальных размеров кадра
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video file")
        return

    # Получение размеров экрана
    screen_width = 1920  # Замените на реальную ширину экрана
    screen_height = 1080  # Замените на реальную высоту экрана

    # Масштабирование кадра по ширине экрана
    frame_height, frame_width, _ = frame.shape
    scale = screen_width / frame_width
    new_height = int(frame_height * scale)
    new_width = screen_width

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразование кадра в формат, подходящий для модели
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)

        # Обработка результатов
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                class_id = int(box.cls[0])
                confidence = box.conf[0]

                # Получение имени класса
                class_name = model.names[class_id]

                # Выбор цвета для рамки по классу
                color = class_colors.get(class_name, (0, 255, 0))  # Зеленый по умолчанию

                # Рисование рамки и текста на кадре
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Масштабирование кадра
        frame_resized = cv2.resize(frame, (new_width, new_height))

        # Отображение обработанного кадра
        cv2.imshow('Real-time Detection', frame_resized)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

# Путь к видеофайлу
video_path = 'D:\\1.mp4'

# Обработка видео в реальном времени
process_video_realtime(video_path)