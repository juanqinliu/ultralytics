from ultralytics import RTDETR, YOLO
from ultralytics.trackers.deepsort import DeepsortTracker
import cv2
import os
import numpy as np
from typing import Literal, Tuple
import argparse
import time

class Tracker:
    def __init__(
        self, 
        model_path: str,
        tracker_type: Literal["bytetrack", "botsort", "deepsort"] = "bytetrack",
        conf_thres: float = 0.5,
        track_id: str = "track1",
        imgsz: Tuple[int, int] = (384, 640)
    ):
        self.model = RTDETR(model_path)
        # self.model = YOLO(model_path)
        self.tracker_type = tracker_type
        self.conf_thres = conf_thres
        self.track_id = track_id
        self.imgsz = imgsz
        if tracker_type == "deepsort":
            self.tracker = DeepsortTracker(self.model, conf_thres=self.conf_thres)

    def track_video(
        self,
        video_path: str,
        save_dir: str = "Track_Results",
        show: bool = True,
        save_frames: bool = True
    ):
        # Create base directory (results/tracker_type)
        base_dir = os.path.join(save_dir, self.tracker_type)
        os.makedirs(base_dir, exist_ok=True)

        # Get next track number
        existing_tracks = [d for d in os.listdir(base_dir) 
                        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('track')]
        next_track_num = len(existing_tracks) + 1

        # Get video name without extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        if self.tracker_type == "deepsort":
            # For deepsort, create track directory and save there
            track_dir = os.path.join(base_dir, f"track{next_track_num}")
            os.makedirs(track_dir, exist_ok=True)
            save_path = os.path.join(track_dir, f"{video_name}.avi")
            
            self.tracker.track_video(
                video_path=video_path,
                show=show,
                save_path=save_path,
                imgsz=self.imgsz
            )
        else:
            # For bytetrack/botsort, save directly to the track directory
            tracker_config = f"{self.tracker_type}.yaml"
            
            results = self.model.track(
                source=video_path,
                imgsz=self.imgsz,
                show=show,
                tracker=tracker_config,
                save=save_frames,
                project=base_dir,  
                name=f"track{next_track_num}",  # 直接使用track号作为name
                conf=self.conf_thres,
                exist_ok=True,
                verbose=True,
                stream=True
            )
            
            for result in enumerate(results):
                if hasattr(result, 'speed'):
                    speed = result.speed

def parse_args():
    parser = argparse.ArgumentParser(description='Object Tracking with Different Trackers')
    parser.add_argument('--model-path', type=str, default="runs/detect/train/weights/best.pt", help='Path to the YOLO model')
    parser.add_argument('--video-path', type=str, default="/home/ljq/UAV-Tracking/Dataset/video/DJI_0003_D_S_E_test.mp4", help='Path to the input video')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=[384,640], help='inference size (pixels)')
    parser.add_argument('--tracker-type', type=str, default='deepsort', choices=['botsort', 'bytetrack', 'deepsort'], help='Type of tracker to use')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--save-dir', type=str, default='Track_Results', help='Directory to save results')
    parser.add_argument('--show', action='store_true', default=True, help='Show tracking results in real-time')
    parser.add_argument('--save-frames', action='store_true', default=True, help='Save tracked frames')
    parser.add_argument('--track-id', type=str, default='track1', help='Track identifier (e.g., track1, track2)')
    return parser.parse_args()

def main():
    args = parse_args()
    tracker = Tracker(
        model_path=args.model_path,
        tracker_type=args.tracker_type,
        conf_thres=args.conf_thres,
        track_id=args.track_id,
        imgsz=tuple(args.imgsz)
    )
    
    tracker.track_video(
        video_path=args.video_path,
        save_dir=args.save_dir,
        show=args.show,
        save_frames=args.save_frames
    )

if __name__ == "__main__":
    main()