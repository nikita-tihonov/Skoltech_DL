from ultralytics import YOLO


model = YOLO("yolo11m.pt")
result = model.train(data="rdd_dataset_japan.yaml", epochs=100, imgsz=640, iou=0.7)
