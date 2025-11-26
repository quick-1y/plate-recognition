"""Plate recognition utilities using EasyOCR and regex pattern filtering."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List

import cv2
import easyocr
import yaml

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PlateRecognizer:
    """Encapsulates plate preprocessing and text recognition logic."""

    def __init__(
        self,
        patterns_path: Path | str = Path("configs/plate_patterns.yaml"),
        languages: Iterable[str] | None = None,
        use_gpu: bool = False,
        debug_dir: Path | None = None,
    ) -> None:
        self.patterns_path = Path(patterns_path)
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.reader = self._create_reader(languages, use_gpu)
        self.patterns = self._load_patterns()

    def _create_reader(self, languages: Iterable[str] | None, use_gpu: bool) -> easyocr.Reader:
        languages = list(languages) if languages else ["en"]
        try:
            return easyocr.Reader(languages, gpu=use_gpu)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to initialize EasyOCR with GPU=%s. Falling back to CPU. Error: %s", use_gpu, exc)
            return easyocr.Reader(languages, gpu=False)

    def _load_patterns(self) -> List[dict]:
        if not self.patterns_path.exists():
            LOGGER.warning("Pattern file %s not found. No pattern filtering will be applied.", self.patterns_path)
            return []

        with self.patterns_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        patterns = config.get("plate_patterns", [])
        if not patterns:
            LOGGER.warning("No plate patterns found in %s.", self.patterns_path)
        return patterns

    def recognize_plate(self, img_gray) -> str:
        """Recognize a plate number from a preprocessed grayscale image."""
        recognized_text = self.recognize_text(img_gray)
        filtered_text = self.filter_by_pattern(recognized_text)
        return filtered_text

    def filter_by_pattern(self, text: str) -> str:
        text = text.upper()
        for pattern in self.patterns:
            regex = re.compile(pattern.get("pattern", ""))
            if regex.match(text):
                return f"{text} {pattern.get('region', '')}".strip()
        return ""

    def preprocess_image(self, img):
        """Resize, denoise and convert the image to grayscale for OCR."""
        self._save_debug_image(img, "debug_original_image.png")

        resized = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.convertScaleAbs(img_gray, alpha=1.5, beta=0)
        img_gray = cv2.bilateralFilter(img_gray, 9, 75, 75)

        self._save_debug_image(img_gray, "debug_preprocessed_image.png")
        return img_gray

    def recognize_text(self, img_gray) -> str:
        results = self.reader.readtext(img_gray, detail=0)
        return "".join(results)

    def _save_debug_image(self, img, filename: str) -> None:
        if not self.debug_dir:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.debug_dir / filename), img)
