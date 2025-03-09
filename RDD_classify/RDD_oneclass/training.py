from ultralytics import YOLO


model = YOLO("yolo11s.pt")
result = model.train(data="rdd_dataset.yaml", epochs=100, imgsz=640, iou=0.7)
