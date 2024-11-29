from ultralytics import YOLO
import torch

# 加载模型
model = YOLO("runs/mbyolo/motfly18/weights/best.pt")
# 导出模型
model.export(format="onnx",dynamic=True)
