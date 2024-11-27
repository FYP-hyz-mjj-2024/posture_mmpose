import time

import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

from step03_yolo_phone_detection.dvalue import best_pt_path
import matplotlib.pyplot as plt

# 检测设备
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


if __name__ == "__main__":
    pt_path = best_pt_path
    # 加载模型
    model = YOLO(pt_path)

    image_path = "./Phone-detection-2/valid/images/10005_jpeg.rf.ebea88d5706bf8903efef9acdc6ce895.jpg"
    image = Image.open(image_path)
    width, height = image.size


    numpy_image = np.array(image)

    # PIL
    start_PIL = time.time()
    print('=======PIL======')
    for _ in range(2000):

        image_PIL = Image.fromarray(numpy_image)
        results_PIL = model(image_PIL, device=device)[0]

    end_PIL = time.time()
    PIL_use_time = end_PIL - start_PIL
    print(f'========PIL use time: {end_PIL - start_PIL} ========')

    # Tensor
    start_tensor = time.time()
    print('=====Tensor=====')
    for _ in range(2000):
        tensor_image = torch.from_numpy(numpy_image).float() / 255.0
        tensor_image = tensor_image.permute(2,0,1).unsqueeze(0).to(device)
        results_tensor = model(tensor_image, device=device)[0]

    end_tensor = time.time()
    tensor_use_time = end_tensor - start_tensor
    print(f'========Tensor use time: {end_tensor - start_tensor} ========\n')



    print(f"tensor use: {tensor_use_time} PIL use: {PIL_use_time}")

    # print(results_PIL.names)
    #
    # print(results_PIL.boxes.xyxyn)
    #
    # print(results_PIL.boxes.cls)
    #
    # print(results_PIL.boxes.conf)