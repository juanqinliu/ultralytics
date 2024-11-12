# rtdetr_track.py
from ultralytics import RTDETR
import cv2
import torch
import numpy as np
from ultralytics.trackers.deep_sort.utils.parser import get_config
from ultralytics.trackers.deep_sort.deep_sort.deep_sort import DeepSort
from collections import deque

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

    def process_detections(self, results):
        """处理检测结果，返回DeepSORT所需的输入格式"""
        if len(results.boxes) == 0:
            return None, None, None
            
        # 获取所有边界框的坐标
        boxes = results.boxes.xyxy.cpu().numpy()
        
        # 计算中心点和宽高
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        x_center = boxes[:, 0] + width/2
        y_center = boxes[:, 1] + height/2
        
        # 构建xywh数组
        xywhs = np.stack([x_center, y_center, width, height], axis=1)
        
        # 获取置信度和类别
        confs = results.boxes.conf.cpu().numpy()
        oids = results.boxes.cls.cpu().numpy()
        
        # 转换为tensor
        xywhs = torch.from_numpy(xywhs)
        confs = torch.from_numpy(confs)
        
        return xywhs, confs, oids

    def draw_boxes(self, img, bbox, identities=None, object_ids=None):
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
            label = f"ID: {id}"
            if object_ids is not None:
                label += f" Class: {self.model.names[object_ids[i]]}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)
            
            # # 绘制轨迹
            # for j in range(1, len(self.data_deque[id])):
            #     if self.data_deque[id][j-1] is None or self.data_deque[id][j] is None:
            #         continue
            #     thickness = int(np.sqrt(64/float(j+1))*2)
            #     cv2.line(img, self.data_deque[id][j-1], self.data_deque[id][j], color, thickness)
                
        return img

    def track_video(self, video_path, show=True, save_path=None):
        """处理视频文件"""
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 设置视频保存
        if save_path:
            out = cv2.VideoWriter(save_path, 
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                fps, 
                                (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # RTDETR检测
            results = self.model.predict(frame, conf=self.conf_thres, verbose=False)[0]
            
            if len(results.boxes) > 0:
                # 处理检测结果
                xywhs, confs, oids = self.process_detections(results)
                
                if xywhs is not None:
                    # DeepSORT更新
                    outputs = self.deepsort.update(xywhs, confs, oids, frame)
                    
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -2]
                        object_ids = outputs[:, -1]
                        
                        # 绘制跟踪结果
                        frame = self.draw_boxes(frame, bbox_xyxy, identities, object_ids)
            
            if show:
                cv2.imshow('DeepSORT Tracking', frame)
                
            if save_path:
                out.write(frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        if save_path:
            out.release()
        cv2.destroyAllWindows()


