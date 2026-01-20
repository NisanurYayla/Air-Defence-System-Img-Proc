from inference.model_loader import DetectorModel
#load_model fonk test

detector= DetectorModel("models/yolo11n.pt")
if not detector.load_model():
    exit(1)