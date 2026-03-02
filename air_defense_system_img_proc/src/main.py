import time
import os
import cv2
from ultralytics import YOLO

from inference.detector import Detector
from inputs.video_input import VideoInput
from visualization.visualizer import draw_detections, draw_tracks, draw_telemetry
from tracking.tracker import Tracker


def main():
    print("[INFO] Sistem başlatıldı")

    # -------------------------
    # MODEL
    # -------------------------
    model = YOLO("models/yolo11s.pt")  # s modeli daha stabil ID veriyor

    # -------------------------
    # DETECTOR
    # -------------------------
    detector = Detector(
        model=model,
        target_fps=20.0
    )

    # -------------------------
    # TRACKER
    # -------------------------
    tracker = Tracker()

    # -------------------------
    # INPUT
    # -------------------------
    input_source = VideoInput("data/videos/test6.mp4")   #6
    input_source.open()

    frame_id = 0
    CONF_THRESHOLD = 0.45  # çok düşük yapma
    FIXED_SIZE = 640       # resolution sabit

    try:
        while True:
            frame = input_source.read()
            if frame is None:
                print("[INFO] Video bitti")
                break

            frame_id += 1

            # -------------------------
            # SABİT RESOLUTION
            # -------------------------
            frame = input_source.resize(frame, FIXED_SIZE, FIXED_SIZE)

            # -------------------------
            # INFERENCE
            # -------------------------
            detections, system_fps, conf = detector.infer(frame)

            # Confidence filtresi
            filtered_dets = []
            for det in detections:
                if det["confidence"] >= CONF_THRESHOLD:
                    filtered_dets.append(det["bbox"])

            # -------------------------
            # TRACKING
            # -------------------------
            tracks = tracker.update(filtered_dets)

            # -------------------------
            # VISUALIZATION
            # -------------------------
            frame = draw_detections(frame, detections)   # raw detection (isteğe bağlı)
            frame = draw_tracks(frame, tracks)           # confirmed tracks

            frame = draw_telemetry(
                frame,
                fps=system_fps,
                num_dets=len(filtered_dets),
                num_tracks=len(tracks)
            )

            cv2.imshow("Air Defence System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(
                f"[FRAME {frame_id:04d}] "
                f"FPS: {system_fps:5.2f} | "
                f"Dets: {len(filtered_dets)} | "
                f"Tracks: {len(tracks)}"
            )

    finally:
        input_source.close()
        cv2.destroyAllWindows()
        print("[INFO] Sistem kapatıldı")


if __name__ == "__main__":
    main()
