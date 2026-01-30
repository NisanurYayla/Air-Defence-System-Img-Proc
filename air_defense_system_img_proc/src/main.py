import time
import os
from ultralytics import YOLO

from inference.detector import Detector
from inference.performance_manager import PerformanceManager
from inputs.video_input import VideoInput
from visualization.visualizer import draw_detections, draw_telemetry
import cv2



def main():
    print("[INFO] Sistem başlatıldı")
    debug_dir = "data/debug"
    os.makedirs(debug_dir, exist_ok=True)


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

    frame_id = 0
    frame_count = 0
    

    try:
        while True:
            frame = input_source.read()
            if frame is None:
                print("[INFO] Video bitti")
                break
 
            frame_id += 1
            # Resolution adaptasyonu
            target_w, target_h = performance.get_resolution()
            frame = input_source.resize(frame, target_w, target_h)

            # Inference
            detections, system_fps, conf = detector.infer(frame)

            #Visualizer
            frame = draw_detections(frame,detections)
            frame = draw_telemetry(
                frame, 
                fps=system_fps,  #kontrol et
                conf=conf,
                resolution=frame.shape[:2]
            )

            if frame_id % 30 == 0:
                save_path = os.path.join(debug_dir, f"frame_{frame_id:04d}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"[DEBUG] Görüntü kaydedildi: {save_path}")


            # sadece görmek için (WSL'de pencere açmayacağız)
            # cv2.imshow("Air Defence System", frame)

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
