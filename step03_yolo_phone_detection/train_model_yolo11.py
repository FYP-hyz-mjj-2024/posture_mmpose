from roboflow.core.dataset import Dataset
from roboflow.core.version import Version
from ultralytics import YOLO
import torch
import os
import ultralytics
from PIL import Image
import requests

from roboflow import Roboflow, Workspace, Project

from step03_yolo_phone_detection.dvalue import API_KEY_mjj, preset_group

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
preset_group_name = 'low-quality phone detection preset'
preset = preset_group[preset_group_name]

if __name__ == '__main__':
    rf: Roboflow = Roboflow(api_key=API_KEY_mjj)

    workspace: Workspace = rf.workspace(preset['workspace'])
    project: Project = workspace.project(preset['project'])
    version: Version = project.version(preset['version'])
    dataset: Dataset = version.download(preset['dataset'])

    model: YOLO = YOLO('yolo11n.pt')
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
    )

    results = model.val()   # Evaluate the model's performance on the validation set

    success = model.export(format="onnx")   # Export the model to ONNX format