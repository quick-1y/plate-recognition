"""Real-time video processing for license plate detection and recognition."""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PyQt5.QtGui import QImage
from ultralytics import YOLO

from recognition_plate import PlateRecognizer
from utils.config import DEFAULT_CONFIG, load_config

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PRETRAINED_MODEL_PATH = Path("models/best.pt")
CLASS_COLORS = {"licence": (255, 255, 255)}
MODEL = None
PLATE_RECOGNIZER = PlateRecognizer(debug_dir=None)


def _load_model() -> YOLO:
    global MODEL
    if MODEL is None:
        if not PRETRAINED_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model weights not found at {PRETRAINED_MODEL_PATH}")
        MODEL = YOLO(str(PRETRAINED_MODEL_PATH))
    return MODEL


def process_video_realtime(
    video_path: str,
    frame_callback: Callable[[QImage], None],
    text_callback: Callable[[str], None],
    config_path: str | Path = "config.yaml",
) -> None:
    """Process a video file frame-by-frame and emit frames and recognized text."""
    config = load_config(config_path)
    plate_image_send_interval = config.get("plate_image_send_interval", DEFAULT_CONFIG["plate_image_send_interval"])

    track_history = defaultdict(list)
    last_plate_position = {}
    last_recognized_plate = ""

    try:
        model = _load_model()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            LOGGER.error("Could not open video file %s", video_path)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = 1.0 / fps if fps else 0

        frame_counter = 0
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.track(frame_rgb, persist=True)

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    track_id = int(box.id[0]) if box.id is not None else None

                    color = CLASS_COLORS.get(class_name, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{class_name} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                    if class_name == "licence":
                        plate_image = frame[y1:y2, x1:x2]
                        preprocessed_image = PLATE_RECOGNIZER.preprocess_image(plate_image)
                        recognized_text = PLATE_RECOGNIZER.recognize_plate(preprocessed_image)

                        if recognized_text and recognized_text != last_recognized_plate:
                            text_callback(recognized_text)
                            last_recognized_plate = recognized_text

                        cv2.putText(frame, recognized_text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        if track_id is not None:
                            _update_track(track_history, track_id, x1, x2, y1, y2, frame, color)
                            current_position = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            if last_plate_position.get(track_id) != current_position:
                                last_plate_position[track_id] = current_position

                        if frame_counter % plate_image_send_interval == 0:
                            LOGGER.info("Frame %s: recognized plate %s", frame_counter, recognized_text)

            frame_for_gui = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_for_gui.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame_for_gui.data, width, height, bytes_per_line, QImage.Format_RGB888)
            frame_callback(q_img)

            _sync_with_fps(frame_delay, start_time)

        cap.release()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Error during video processing: %s", exc)


def _update_track(track_history, track_id, x1, x2, y1, y2, frame, color):
    track = track_history[track_id]
    current_position = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    track.append(current_position)
    if len(track) > 30:
        track.pop(0)
    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)


def _sync_with_fps(frame_delay: float, start_time: float) -> None:
    if frame_delay <= 0:
        return
    elapsed_time = time.time() - start_time
    if elapsed_time < frame_delay:
        time.sleep(frame_delay - elapsed_time)
