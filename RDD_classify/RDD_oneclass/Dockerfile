FROM ultralytics/ultralytics

COPY training.py .
COPY rdd_dataset.yaml .
COPY China_MotorBike /China_MotorBike

ENTRYPOINT ["python3", "training.py"]
