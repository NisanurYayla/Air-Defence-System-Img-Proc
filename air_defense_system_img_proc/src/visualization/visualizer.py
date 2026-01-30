#Visualizer modülü, algılama (detection) katmanından gelen çıktıları gerçek zamanlı olarak
#  görselleştirmek için kullanılır. Algılanan nesnelerin bounding box’larını, güven skorlarını
#  ve sistem telemetri bilgilerini (FPS, çözünürlük, confidence) video karesi üzerine çizer. 
# Karar verme veya performans yönetimi içermez; yalnızca debug, analiz ve sistemin davranışını 
# gözle doğrulamak amacıyla kullanılır

import cv2
import numpy as np
from typing import List, Dict, Tuple


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict]
) -> np.ndarray:
    """
    Algılanan nesnelerin bounding box ve etiketlerini çizer.
    """

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["confidence"]
        cls_id = det["class_id"]

        label = f"ID:{cls_id} {conf:.2f}"

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return frame


def draw_telemetry(
    frame: np.ndarray,
    fps: float,
    conf: float,
    resolution: Tuple[int, int]
) -> np.ndarray:
    """
    Sistem telemetri bilgilerini ekrana yazar.
    """

    h, w = resolution
    text = f"FPS: {fps:.1f} | Conf: {conf:.2f} | Res: {w}x{h}"

    cv2.putText(
        frame,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )

    return frame
