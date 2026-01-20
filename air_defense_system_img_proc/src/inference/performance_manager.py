import time
from typing import Tuple, List


class PerformanceManager:
    """
    Performans ve görev durumu farkındalıklı adaptasyon modülü.
    - FPS trendine göre resolution ayarlar
    - Lock-on sırasında resolution'u kilitler
    """

    def __init__(
        self,
        target_fps: float = 20.0,
        fps_margin: float = 3.0,
        low_fps_frames: int = 5,
        high_fps_frames: int = 10,
        cooldown_sec: float = 1.5
    ):
        self.target_fps = target_fps
        self.fps_margin = fps_margin
        self.low_fps_frames = low_fps_frames
        self.high_fps_frames = high_fps_frames
        self.cooldown_sec = cooldown_sec

        # Resolution seviyeleri (LOW → HIGH)
        self.levels: List[Tuple[int, int]] = [
            (416, 416),
            (512, 512),       #32 nin katları 
            (640, 640)
        ]

        self.current_level: int = 2  # HIGH
        self.low_fps_counter: int = 0
        self.high_fps_counter: int = 0

        self.last_change_time: float = time.time()

        # Görev durumu
        self.lock_on: bool = False

    # ---------- PUBLIC API ----------

    def update(self, system_fps: float) -> None:
        """
        Detector'dan gelen FPS ile state güncelle.
        """

        if self.lock_on:
            return  # Lock-on sırasında resolution sabit

        now = time.time()
        if now - self.last_change_time < self.cooldown_sec:
            return

        if system_fps < self.target_fps:
            self.low_fps_counter += 1
            self.high_fps_counter = 0
        elif system_fps > self.target_fps + self.fps_margin:
            self.high_fps_counter += 1
            self.low_fps_counter = 0
        else:
            self.low_fps_counter = 0
            self.high_fps_counter = 0

        if self.low_fps_counter >= self.low_fps_frames:
            self._decrease_resolution()

        if self.high_fps_counter >= self.high_fps_frames:
            self._increase_resolution()

    def get_resolution(self) -> Tuple[int, int]:
        return self.levels[self.current_level]

    def set_lock_on(self, lock: bool) -> None:
        """
        Üst seviye karar modülü tarafından çağrılır.
        """
        self.lock_on = lock

    # ---------- INTERNAL ----------

    def _decrease_resolution(self) -> None:
        if self.current_level > 0:
            self.current_level -= 1
            self._reset_state()

    def _increase_resolution(self) -> None:
        if self.current_level < len(self.levels) - 1:
            self.current_level += 1
            self._reset_state()

    def _reset_state(self) -> None:
        self.low_fps_counter = 0
        self.high_fps_counter = 0
        self.last_change_time = time.time()
