"""Application window for license plate recognition."""
from __future__ import annotations

import sys

from gui.common import (
    QApplication,
    QFileDialog,
    QDesktopWidget,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QImage,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QPixmap,
    QThread,
    pyqtSignal,
)
from process_video_realtime import process_video_realtime
from utils.config import load_config, save_config


class VideoThread(QThread):
    frame_signal = pyqtSignal(QImage)
    text_signal = pyqtSignal(str)

    def __init__(self, video_path: str, config_path: str):
        super().__init__()
        self.video_path = video_path
        self.config_path = config_path

    def run(self):
        try:
            process_video_realtime(self.video_path, self.frame_signal.emit, self.text_signal.emit, self.config_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Error during video processing: {exc}")


class MainWindow(QMainWindow):
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__()
        self.config_path = config_path
        self.recognized_plates = set()
        self.video_threads: list[VideoThread] = []
        self.config = load_config(self.config_path)
        self.initUI()
        self.start_processing()

    def update_text(self, text: str):
        if text and text not in self.recognized_plates:
            item = QListWidgetItem(text)
            self.events_list.addItem(item)
            self.recognized_plates.add(text)

    def initUI(self):
        self.setWindowTitle("Программа для распознования автомобильных номеров")
        screen = QDesktopWidget().screenGeometry()
        window_width = int(screen.width() * 0.8)
        window_height = int(screen.height() * 0.8)
        self.setGeometry((screen.width() - window_width) // 2, (screen.height() - window_height) // 2, window_width, window_height)

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        video_and_list_layout = QHBoxLayout()
        main_layout.addLayout(video_and_list_layout)

        self.video_label = QLabel(self)
        video_width = int(window_width * 0.7)
        video_height = int(window_height * 0.7)
        self.video_label.setFixedSize(video_width, video_height)
        self.video_label.setScaledContents(True)
        video_and_list_layout.addWidget(self.video_label)

        self.tab_widget = QTabWidget(self)
        list_width = int(window_width * 0.3)
        list_height = int(window_height * 0.7)
        self.tab_widget.setFixedSize(list_width, list_height)
        video_and_list_layout.addWidget(self.tab_widget)

        self.events_list = QListWidget(self)
        self.tab_widget.addTab(self.events_list, "События")

        self.search_list = QListWidget(self)
        self.tab_widget.addTab(self.search_list, "Поиск")

        self.lists_list = QListWidget(self)
        self.tab_widget.addTab(self.lists_list, "Списки")

        self.settings_tab = QWidget()
        self.init_settings_tab()
        self.tab_widget.addTab(self.settings_tab, "Настройки")

        self.video_thread = None

    def start_processing(self):
        self.stop_processing()
        self.video_threads = []
        for video_path in self.config["video_paths"]:
            if not video_path:
                continue
            video_thread = VideoThread(video_path, self.config_path)
            video_thread.frame_signal.connect(self.update_frame)
            video_thread.text_signal.connect(self.update_text)
            video_thread.start()
            self.video_threads.append(video_thread)

    def stop_processing(self):
        for video_thread in self.video_threads:
            if video_thread and video_thread.isRunning():
                video_thread.terminate()
                video_thread.wait()

    def update_frame(self, q_img: QImage):
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap)

    def init_settings_tab(self):
        layout = QVBoxLayout(self.settings_tab)
        self.settings_tab_widget = QTabWidget(self)
        layout.addWidget(self.settings_tab_widget)

        self.processing_settings_tab = QWidget()
        self.init_processing_settings_tab()
        self.settings_tab_widget.addTab(self.processing_settings_tab, "Настройки обработки видео")

        self.channels_settings_tab = QWidget()
        self.init_channels_settings_tab()
        self.settings_tab_widget.addTab(self.channels_settings_tab, "Настройки каналов")

        save_button = QPushButton("Сохранить настройки", self)
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)

    def init_processing_settings_tab(self):
        layout = QFormLayout(self.processing_settings_tab)
        self.interval_edit = QLineEdit(self)
        self.interval_edit.setText(str(self.config["plate_image_send_interval"]))
        layout.addRow("Отправка изображения каждые (кадров):", self.interval_edit)

    def init_channels_settings_tab(self):
        layout = QVBoxLayout(self.channels_settings_tab)

        form_layout = QFormLayout()
        self.channel_count_edit = QLineEdit(self)
        self.channel_count_edit.setText(str(len(self.config["video_paths"])))
        form_layout.addRow("Количество каналов (не больше 10):", self.channel_count_edit)
        layout.addLayout(form_layout)

        self.video_path_edits = []
        for i in range(10):
            video_path_edit = QLineEdit(self)
            video_path_edit.setReadOnly(True)
            video_path_edit.mousePressEvent = lambda event, idx=i: self.select_video(event, idx)
            self.video_path_edits.append(video_path_edit)
            form_layout.addRow(f"Путь к видео для канала {i + 1}:", video_path_edit)

        for i, path in enumerate(self.config["video_paths"]):
            self.video_path_edits[i].setText(path)

    def select_video(self, event, index: int):
        self.stop_processing()
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi);;All Files (*)",
            options=options,
        )
        if file_name:
            self.video_path_edits[index].setText(file_name)

    def save_settings(self):
        new_interval = self.interval_edit.text()
        new_channel_count = self.channel_count_edit.text()
        new_video_paths = [edit.text() for edit in self.video_path_edits if edit.text()]

        if new_interval.isdigit():
            self.config["plate_image_send_interval"] = int(new_interval)

        if new_channel_count.isdigit():
            channel_count = int(new_channel_count)
            if 0 < channel_count <= 10:
                self.config["video_paths"] = new_video_paths[:channel_count]
                self.start_processing()

        save_config(self.config_path, self.config)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
