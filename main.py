# from ultralytics import YOLO
from ultralytics import RTDETR
import cv2
import os
import numpy as np

# 加载模型
model = RTDETR("runs/detect/train/weights/best.pt") 
path = model.export(format="onnx", opset=16)

print('Model classes:', model.names)

# 创建保存目录
save_dir = "results" 
os.makedirs(save_dir, exist_ok=True)

# 进行目标检测和追踪
results = model.track(
    source="/home/ljq/UAV-Tracking/Dataset/video/DJI_0003_D_S_E_test.mp4",
    show=True,
    tracker="botsort.yaml",
    save_frames=True,
    save_dir=save_dir,
    conf=0.5  # 设置置信度阈值
)
