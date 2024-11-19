from ultralytics import YOLO
import torch

from step03_yolo_phone_detection.dvalue import yaml_path, single_test

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO("yolo11n.yaml")    # Create a new YOLO model from scratch
model = YOLO("yolo11n.pt")  # Load a pretrained YOLO model (recommended for training)

results = model.train(data=yaml_path, epochs=1, device=device)
results = model.val()   # Evaluate the model's performance on the validation set

results = model(single_test)
success = model.export(format="onnx")   # Export the model to ONNX format