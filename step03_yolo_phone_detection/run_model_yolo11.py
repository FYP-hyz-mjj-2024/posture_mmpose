import torch
from ultralytics import YOLO
from PIL import Image

from step03_yolo_phone_detection.dvalue import best_pt_path

# 检测设备
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

pt_path = best_pt_path
# 加载模型
model = YOLO(pt_path)

image_path = "./Phone-detection-2/valid/images/10005_jpeg.rf.ebea88d5706bf8903efef9acdc6ce895.jpg"
image = Image.open(image_path)
width, height = image.size

# 进行推理
results = model(image, device=device)[0]

print('================')

print(results.names)

print(results.boxes.xyxyn)

print(results.boxes.cls)

print(results.boxes.conf)

print('================')
