# System
import shutil
import time
import os
import json

# Utils
import torch

# Roboflow
from roboflow.core.dataset import Dataset
from roboflow.core.version import Version
from ultralytics import YOLO
from roboflow import Roboflow, Workspace, Project

# Local
from step03_yolo_phone_detection.dvalue import yolo_input_size
from step03_yolo_phone_detection.pvalue import API_KEY_mjj

global_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
preset_group_name = 'yolo dataset'

# Configuration File
config_json = "./config.json"
if not os.path.exists(config_json):
    raise FileNotFoundError("Can not find preset group configuration file config.json."
                            "It should be a .json file at the same level with this file.")


if __name__ == '__main__':
    # Read configuration file and increase the version.
    print("Reading configuration...")
    with open(config_json, "r") as f:
        config = json.load(f)
        f.close()

    # Read the preset, increase version number.
    config["preset_group"][preset_group_name]["version"] += 1
    preset = config["preset_group"][preset_group_name]
    print(f"Current preset version: {preset['version']}")

    # Rename previous train files: Push-down previous best
    time_str = time.strftime("%Y%m%d_%H%M")
    runs_detect = "./runs/detect"
    print(f"Renaming previous train files with time string {time_str}...")
    for item in os.listdir(runs_detect):
        item_path = os.path.join(runs_detect, item)
        if not os.path.isdir(item_path) or item.startswith("_"):
            continue
        new_item_path = os.path.join(runs_detect, f"_{time_str}_{item}")
        os.rename(item_path, new_item_path)

    # Initialize Roboflow datasets and YOLO model
    rf: Roboflow = Roboflow(api_key=API_KEY_mjj)

    workspace: Workspace = rf.workspace(preset['workspace'])
    project: Project = workspace.project(preset['project'])
    version: Version = project.version(preset['version'])
    dataset: Dataset = version.download(model_format=preset['dataset'],
                                        location="./roboflow_dataset_download",
                                        overwrite=True)

    model: YOLO = YOLO('./non_tuned/yolo11n.pt')

    # Model Training & Evaluation
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=100,
        imgsz=yolo_input_size,
        batch=16,
        device=global_device,
    )

    results = model.val()   # Evaluate the model's performance on the validation set

    # Archive the previous best.pt in archived_onnx.
    print("\nArchiving the old best.pt in archived_onnx...", end="")
    old_bestpt = "./archived_onnx/best.pt"
    archived_bestpt = f"./archived_onnx/{time_str}_best.pt"
    shutil.move(old_bestpt, archived_bestpt)
    print(f"Done! The archived previous best.pt is now {time_str}_best.pt")

    # Copy the latest best.pt as the new best into archived_onnx.
    print("\nWelcoming the new best.pt in archived_onnx...", end="")
    new_bestpt_from = "./runs/detect/train/weights/best.pt"
    new_bestpt_to = "./archived_onnx"
    shutil.copy(new_bestpt_from, new_bestpt_to)
    print("Done!")

    # Deploy model to roboflow.
    print("\nDeploying model to Roboflow...", end="")
    project.version(dataset.version).deploy(model_type="yolo11", model_path="./runs/detect/train/")
    print("Done!")

    # Write new version into json.
    print("\nWriting configuration...", end="")
    with open(config_json, "w") as f:
        json.dump(config, f, indent=2)
    print("Done!")

