#Visualizer modülü, algılama (detection) katmanından gelen çıktıları gerçek zamanlı olarak
#  görselleştirmek için kullanılır. Algılanan nesnelerin bounding box’larını, güven skorlarını
#  ve sistem telemetri bilgilerini (FPS, çözünürlük, confidence) video karesi üzerine çizer. 
# Karar verme veya performans yönetimi içermez; yalnızca debug, analiz ve sistemin davranışını 
# gözle doğrulamak amacıyla kullanılır
import cv2
import numpy as np


# -------------------------------------------------
# DRAW RAW DETECTIONS (isteğe bağlı)
# -------------------------------------------------
def draw_detections(frame, detections):
    """
    detections: list of dict
    {
        "bbox": [x1,y1,x2,y2],
        "confidence": float,
        "class_id": int
    }
    """

    for det in detections:
        bbox = det["bbox"]
        conf = det["confidence"]

        x1, y1, x2, y2 = map(int, bbox)

        color = (100, 100, 100)  # gri
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        label = f"{conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA
        )

    return frame


# -------------------------------------------------
# DRAW TRACKS (CONFIRMED ONLY)
# -------------------------------------------------
def draw_tracks(frame, tracks):
    """
    tracks: list of Track objects (CONFIRMED)
    """

    for track in tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = map(int, bbox)

        # Ana kutu (yeşil)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ID etiketi
        label = f"ID {track.id}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

        # Merkez noktası
        cx, cy = map(int, track.center)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        # Velocity vektörü (küçük ok)
        vx, vy = track.velocity
        end_point = (int(cx + vx * 5), int(cy + vy * 5))

        cv2.arrowedLine(
            frame,
            (cx, cy),
            end_point,
            (255, 0, 0),
            2,
            tipLength=0.4
        )

        # Trail (son 10 merkez)
        for i in range(1, len(track.history)):
            pt1 = tuple(map(int, track.history[i - 1]))
            pt2 = tuple(map(int, track.history[i]))
            cv2.line(frame, pt1, pt2, (0, 150, 255), 1)

    return frame


# -------------------------------------------------
# TELEMETRY PANEL
# -------------------------------------------------
def draw_telemetry(frame, fps, num_dets, num_tracks):

    h, w = frame.shape[:2]

    panel_height = 70
    overlay = frame.copy()

    # Alt panel
    cv2.rectangle(
        overlay,
        (0, h - panel_height),
        (w, h),
        (30, 30, 30),
        -1
    )

    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Metinler
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Detections: {num_dets}",
        (200, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Active Tracks: {num_tracks}",
        (420, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return frame
