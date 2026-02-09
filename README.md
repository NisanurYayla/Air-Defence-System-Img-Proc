Air Defense System – Image Processing Module
📌 Project Overview

This project implements a real-time image processing pipeline developed for an air defense simulation system.
The module processes video input streams, detects aerial threats using AI-based object detection, tracks detected
targets across frames, and visualizes system outputs for debugging and monitoring.
The system is designed with a modular architecture to support scalability, hardware portability (PC / Jetson), and real-time performance


System Pipeline

The processing flow of the system is:
Video Input
    ↓
Preprocessing / Resize
    ↓
Object Detection (AI Model)
    ↓
Tracking
    ↓
Visualization & Debug Output



🧩 Module Descriptions
🎥 Input Module

Responsible for receiving video streams from:
  Video files
  Camera devices
  Future support for real-time camera feeds

Handles:
  Frame reading
  Resolution resizing
  Source management



🧠 Detection Module

Uses AI-based object detection to identify aerial targets.

Features:

YOLO-based detection model
Confidence filtering
Bounding box extraction
Class prediction
Output example:

{
    "bbox": (x1, y1, x2, y2),
    "confidence": 0.85,
    "class_id": 0
}



🖥️ Visualization Module

Displays system outputs for debugging and monitoring.

Features:

Bounding box drawing
Target ID display
Confidence visualization
Debug frame saving



📊 Performance Manager

Tracks real-time performance metrics such as:

FPS
Frame count
Processing time
Helps optimize real-time system behavior.



🚀 How to Run
1️⃣ Activate Virtual Environment
source venv/bin/activate

2️⃣ Run the System
python3 src/main.py



📺 Output

The system prints real-time processing information:
[FRAME 0123] FPS: 82.45 | Detections: 2 | Resolution: 640x640


Debug frames can also be saved to:
debug/
