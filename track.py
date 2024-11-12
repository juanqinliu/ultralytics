from ultralytics import RTDETR
from ultralytics.trackers.deepsort import DeepsortTracker
import cv2
import os
import numpy as np
from typing import Literal
import argparse

class Tracker:
    def __init__(
        self, 
        model_path: str,
        tracker_type: Literal["bytetrack", "botsort", "deepsort"] = "bytetrack",
        conf_thres: float = 0.5
    ):
        self.model = RTDETR(model_path)
        self.tracker_type = tracker_type
        self.conf_thres = conf_thres
        
        if tracker_type == "deepsort":
            self.tracker = DeepsortTracker(self.model, conf_thres=self.conf_thres)
    
    def track_video(
        self,
        video_path: str,
        save_dir: str = "results",
        show: bool = True,
        save_frames: bool = True
    ):
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        if self.tracker_type == "deepsort":
            # deepsort
            save_path = os.path.join(save_dir, "tracked_video.mp4")
            results = self.tracker.track_video(
                video_path=video_path,
                show=show,
                save_path=save_path
            )
        else:
            # bytetrack or botsort
            tracker_config = f"{self.tracker_type}.yaml"
            results = self.model.track(
                source=video_path,
                show=show,
                tracker=tracker_config,
                save_frames=save_frames,
                save_dir=save_dir,
                conf=self.conf_thres
            )
        return results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Object Tracking with Different Trackers')
    parser.add_argument('--model-path', type=str, default="runs/detect/train/weights/best.pt", help='Path to the RTDETR model')
    parser.add_argument('--video-path', type=str, default="/home/ljq/UAV-Tracking/Dataset/video/DJI_0003_D_S_E_test.mp4", help='Path to the input video')
    parser.add_argument('--tracker-type', type=str,default='deepsort', choices=['bytetrack', 'botsort', 'deepsort'],help='Type of tracker to use')
    parser.add_argument('--conf-thres', type=float,default=0.5,help='Confidence threshold')
    parser.add_argument('--save-dir', type=str,default='results',help='Directory to save results')
    parser.add_argument('--show', default=True, action='store_true',help='Show tracking results in real-time')
    parser.add_argument('--save-frames', action='store_true',help='Save tracked frames')
    return parser.parse_args()

def main():
    args = parse_args()
    # 创建跟踪器
    tracker = Tracker(
        model_path=args.model_path,
        tracker_type=args.tracker_type,
        conf_thres=args.conf_thres
    )
    # 设置保存目录
    save_dir = os.path.join(args.save_dir, args.tracker_type)
    
    # 执行视频追踪
    tracker.track_video(
        video_path=args.video_path,
        save_dir=save_dir,
        show=args.show,
        save_frames=args.save_frames
    )

if __name__ == "__main__":
    main()