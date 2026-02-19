import time
from typing import List, Dict, Tuple
import numpy as np


class Detector:
    """
    Algı (Perception) katmanı.
    - Frame alır
    - Inference yapar
    - Sistem FPS ölçer
    - Dynamic confidence uygular
    """

    def __init__(
        self,
        model,
        target_fps: float = 20.0,
        conf_init: float = 0.5,
        conf_min: float = 0.4,
        conf_max: float = 0.7,
        conf_step_up: float = 0.05,
        conf_step_down: float = 0.02,
        avg_window: int = 10
    ):
        self.model = model

        # FPS hedefleri
        self.target_fps = target_fps

        # Confidence ayarları
        self.conf = conf_init
        self.conf_min = conf_min
        self.conf_max = conf_max
        self.conf_step_up = conf_step_up
        self.conf_step_down = conf_step_down

        # FPS ölçümü (system FPS)
        self.fps_history: List[float] = []
        self.avg_window = avg_window
        self.system_fps: float = 0.0

        # Telemetry
        self.last_resolution: Tuple[int, int] | None = None

    def infer(self, frame: np.ndarray) -> Tuple[List[Dict], float, float]:
        """
        Tek frame inference.

        Returns:
            detections : list of dict
            system_fps : float
            conf       : float
        """

        system_start = time.time()
        self.last_resolution = frame.shape[:2]  # (H, W)

        # YOLO inference
        results = self.model(
            frame,
            conf=self.conf,
            iou=0.5,
            verbose=False
        )

        detections = self._parse_results(results)

        system_end = time.time()
        system_time = system_end - system_start

        self._update_system_fps(system_time)
        self._adapt_confidence()

        return detections, self.system_fps, self.conf

    def _update_system_fps(self, system_time: float) -> None:
        if system_time <= 0:
            return

        fps = 1.0 / system_time
        self.fps_history.append(fps)

        if len(self.fps_history) > self.avg_window:
            self.fps_history.pop(0)

        self.system_fps = sum(self.fps_history) / len(self.fps_history)

    def _adapt_confidence(self) -> None:
        """
        Değişebilir FPS
        FPS'e göre confidence stabilizasyonu.
        Resolution adaptasyonu bu sınıfta yapılmaz. Bunu performance_manager.py den alıp burada kullanıyoruz.
        """

        if self.system_fps < self.target_fps:
            self.conf = min(self.conf + self.conf_step_up, self.conf_max)
        else:
            self.conf = max(self.conf - self.conf_step_down, self.conf_min)

    def _parse_results(self, results) -> List[Dict]:
        """
        YOLO çıktısını CPU tarafında sadeleştirir.
        """

        detections: List[Dict] = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().item())
                cls = int(box.cls[0].cpu().item())

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "class_id": cls
                })

        return detections
