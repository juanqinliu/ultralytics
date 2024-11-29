# rtdetr_track.py
from ultralytics import RTDETR
import cv2
import os
import torch
import numpy as np
from ultralytics.trackers.deep_sort.utils.parser import get_config
from ultralytics.trackers.deep_sort.deep_sort.deep_sort import DeepSort

from collections import deque
import time

class DeepsortTracker:
    def __init__(self, model, conf_thres=0.3):
        # 初始化RTDETR模型
        self.model = model
        self.conf_thres = conf_thres
        self.data_deque = {}
        
        # 初始化DeepSORT
        cfg_deep = get_config()
        cfg_deep.merge_from_file("ultralytics/cfg/trackers/deepsort.yaml")
        self.deepsort = DeepSort(
            cfg_deep.DEEPSORT.REID_CKPT,
            max_dist=cfg_deep.DEEPSORT.MAX_DIST,
            min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg_deep.DEEPSORT.MAX_AGE,
            n_init=cfg_deep.DEEPSORT.N_INIT,
            nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
            use_cuda=True
        )

    def process_detections(self, boxes, confs, cls):
        """处理检测结果，返回DeepSORT所需的输入格式"""
        if len(boxes) == 0:
            return None
        
        # Move tensors to CPU first
        boxes = boxes.cpu()
        confs = confs.cpu()
        cls = cls.cpu()
        
        # 计算中心点和宽高
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        x_center = boxes[:, 0] + width/2
        y_center = boxes[:, 1] + height/2
        
        # 构建xywh数组
        xywhs = np.stack([x_center.numpy(), y_center.numpy(), 
                        width.numpy(), height.numpy()], axis=1)
        
        # 转换为tensor
        xywhs = torch.from_numpy(xywhs)
        
        return xywhs, confs, cls

    def draw_boxes(self, img, bbox, identities=None, object_ids=None, confs=None):
        """绘制跟踪框和轨迹"""
        height, width = img.shape[:2]
        
        # 清理已经消失的目标的轨迹数据
        for key in list(self.data_deque):
            if identities is None or key not in identities:
                self.data_deque.pop(key)

        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            
            # 计算中心点
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # 获取跟踪ID
            id = int(identities[i]) if identities is not None else 0
            
            # 为新的目标创建轨迹缓冲
            if id not in self.data_deque:
                self.data_deque[id] = deque(maxlen=64)
                
            # 添加中心点到轨迹
            self.data_deque[id].appendleft(center)
            
            # 绘制边界框
            color = (255, 0, 0)   ## 蓝色
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
            
            # 添加ID标签
            label = f"id:{id}"
            if object_ids is not None:
                label += f" {self.model.names[object_ids[i]]}"

            # 添加置信度分数
            if confs is not None:
                label += f" {confs[i]:.2f}"  # 显示2位小数

            # 获取文本大小
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 5)
            
            # 绘制蓝色背景矩形
            background_top_left = (x1, y1 - text_height - 10)  # 文本背景的上左角
            background_bottom_right = (x1 + text_width, y1)   # 文本背景的下右角
            cv2.rectangle(img, background_top_left, background_bottom_right, color, -1)  # 填充矩形

            # 绘制白色字体
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)

            # # 绘制轨迹
            # for j in range(1, len(self.data_deque[id])):
            #     if self.data_deque[id][j-1] is None or self.data_deque[id][j] is None:
            #         continue
            #     thickness = int(np.sqrt(64/float(j+1))*2)
            #     cv2.line(img, self.data_deque[id][j-1], self.data_deque[id][j], color, thickness)
            
        return img

    def track_video(self, video_path, show=True, save_path=None, imgsz=(384, 640)):
            
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                out = cv2.VideoWriter(save_path, 
                                    cv2.VideoWriter_fourcc(*'mp4v'), 
                                    fps, 
                                    (width, height))

            frame_idx = 0
            preprocess_times = []
            inference_times = []
            postprocess_times = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                t1 = time.time()
                # 预处理
                frame_resized = cv2.resize(frame, imgsz)
                t2 = time.time()
                
                # 推理
                results = self.model.predict(frame_resized, conf=self.conf_thres, verbose=False)[0]
                t3 = time.time()
                
                # Scale detection boxes if any exist
                if len(results.boxes) > 0:
                    scale_x = width / imgsz[0]
                    scale_y = height / imgsz[1]
                    # 创建新的 boxes 张量并缩放
                    scaled_boxes = results.boxes.xyxy.clone()
                    scaled_boxes[:, [0, 2]] *= scale_x
                    scaled_boxes[:, [1, 3]] *= scale_y
                    
                    # 处理缩放后的检测结果
                    xywhs = self.process_detections(scaled_boxes, results.boxes.conf, results.boxes.cls)
                    
                    if xywhs is not None:
                        # 创建一个字典来存储每个检测框的置信度
                        conf_dict = {}
                        # 将当前检测结果的坐标和置信度配对存储
                        for i, box in enumerate(xywhs[0]):
                            conf_dict[tuple(box.tolist())] = float(xywhs[1][i])
                        
                        outputs = self.deepsort.update(xywhs[0], xywhs[1], xywhs[2], frame)
                        if len(outputs) > 0:
                            bbox_xyxy = outputs[:, :4]
                            identities = outputs[:, -2]
                            object_ids = outputs[:, -1]
                            
                            # 获取每个跟踪框对应的置信度
                            conf_scores = []
                            for bbox in bbox_xyxy:
                                # 找到最匹配的检测框
                                min_dist = float('inf')
                                matched_conf = 0.0
                                for det_box, conf in conf_dict.items():
                                    # 计算中心点距离
                                    det_center = ((det_box[0] + det_box[2])/2, (det_box[1] + det_box[3])/2)
                                    track_center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                                    dist = ((det_center[0] - track_center[0])**2 + 
                                        (det_center[1] - track_center[1])**2)**0.5
                                    if dist < min_dist:
                                        min_dist = dist
                                        matched_conf = conf
                                conf_scores.append(matched_conf)
                            
                            frame = self.draw_boxes(frame, bbox_xyxy, identities, object_ids, conf_scores)

                t4 = time.time()
                
                # 计算各阶段时间
                preprocess_time = (t2 - t1) * 1000
                inference_time = (t3 - t2) * 1000
                postprocess_time = (t4 - t3) * 1000
                
                preprocess_times.append(preprocess_time)
                inference_times.append(inference_time)
                postprocess_times.append(postprocess_time)
                
                total_time = preprocess_time + inference_time + postprocess_time
                
                # 动态生成检测结果描述
                if len(results.boxes) > 0:
                    class_counts = {}
                    for cls_idx in results.boxes.cls.cpu().numpy():
                        cls_name = self.model.names[int(cls_idx)]
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    det_description = ", ".join([f"{count} {cls_name}" for cls_name, count in class_counts.items()])
                else:
                    det_description = "no objects"
                
                print(f"video 1/1 (frame {frame_idx+1}/{total_frames}) {video_path}: "
                    f"{imgsz[0]}x{imgsz[1]} {det_description}, {total_time:.1f}ms")
                if show:
                    cv2.imshow('Tracking', frame)
                if save_path:
                    out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                frame_idx += 1

            # 打印速度统计
            if frame_idx > 0:
                avg_preprocess = np.mean(preprocess_times)
                avg_inference = np.mean(inference_times)
                avg_postprocess = np.mean(postprocess_times)
                print(f"Speed: {avg_preprocess:.1f}ms preprocess, {avg_inference:.1f}ms inference, {avg_postprocess:.1f}ms postprocess, per image at shape (1, 3, {imgsz[0]}, {imgsz[1]})")

                if save_path:
                    print(f"Result saved video to {save_path}")    
            cap.release()
            if save_path:
                out.release()
            cv2.destroyAllWindows()
