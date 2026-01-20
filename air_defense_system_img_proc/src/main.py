import time
from ultralytics import YOLO

from inference.detector import Detector
from inference.performance_manager import PerformanceManager
from inputs.video_input import VideoInput


def main():
    print("[INFO] Sistem başlatıldı")

    # -------------------------
    # MODEL
    # -------------------------
    model = YOLO("models/yolo11n.pt")

    # -------------------------
    # DETECTOR
    # -------------------------
    detector = Detector(
        model=model,
        target_fps=20.0
    )

    # -------------------------
    # PERFORMANCE MANAGER
    # -------------------------
    performance = PerformanceManager(
        target_fps=20.0
    )

    # -------------------------
    # INPUT (VIDEO)
    # -------------------------
    input_source = VideoInput("data/videos/test.mp4")
    input_source.open()

    frame_count = 0

    try:
        while True:
            frame = input_source.read()
            if frame is None:
                print("[INFO] Video bitti")
                break

            # Resolution adaptasyonu
            target_w, target_h = performance.get_resolution()
            frame = input_source.resize(frame, target_w, target_h)

            # Inference
            detections, system_fps, conf = detector.infer(frame)

            # Performance state update
            performance.update(system_fps)

            # -------------------------
            # LOG (GUI YOK)
            # -------------------------
            frame_count += 1
            print(
                f"[FRAME {frame_count:04d}] "
                f"FPS: {system_fps:5.2f} | "
                f"Conf: {conf:.2f} | "
                f"Detections: {len(detections)} | "
                f"Res: {target_w}x{target_h}"
            )

    finally:
        input_source.close()
        print("[INFO] Sistem kapatıldı")


if __name__ == "__main__":
    main()
