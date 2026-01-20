import cv2
import numpy as np
from inputs.base_input import BaseInput


class VideoInput(BaseInput):
    """
    Video dosyasından frame okur.
    GUI içermez. WSL-safe.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {self.video_path}")

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def resize(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
