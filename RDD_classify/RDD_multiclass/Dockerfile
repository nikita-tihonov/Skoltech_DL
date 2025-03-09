FROM ultralytics/ultralytics

COPY training.py .
COPY rdd_dataset_japan.yaml .
COPY Japan /Japan

ENTRYPOINT ["python3", "training.py"]
