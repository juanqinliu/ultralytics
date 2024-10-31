# Ultralytics YOLO 🚀, AGPL-3.0 license

from functools import partial
from pathlib import Path

import torch
import numpy as np
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker

# A mapping of tracker types to corresponding tracker classes
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}


def on_predict_start(predictor: object, persist: bool = False) -> None:
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool): Whether to persist the trackers if they already exist.

    Raises:
        AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.

    Examples:
        Initialize trackers for a predictor object:
        >>> predictor = SomePredictorClass()
        >>> on_predict_start(predictor, persist=True)
    """
    if hasattr(predictor, "trackers") and persist:
        return

    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))

    if cfg.tracker_type not in {"bytetrack", "botsort"}:
        raise AssertionError(f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'")

    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
        if predictor.dataset.mode != "stream":  # only need one tracker for other modes.
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs  # for determining when to reset tracker on new video


# def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
#     """
#     Postprocess detected boxes and update with object tracking.

#     Args:
#         predictor (object): The predictor object containing the predictions.
#         persist (bool): Whether to persist the trackers if they already exist.

#     Examples:
#         Postprocess predictions and update with tracking
#         >>> predictor = YourPredictorClass()
#         >>> on_predict_postprocess_end(predictor, persist=True)
#     """
#     path, im0s = predictor.batch[:2]

#     is_obb = predictor.args.task == "obb"
#     is_stream = predictor.dataset.mode == "stream"
#     for i in range(len(im0s)):
#         tracker = predictor.trackers[i if is_stream else 0]
#         vid_path = predictor.save_dir / Path(path[i]).name
#         if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
#             tracker.reset()
#             predictor.vid_path[i if is_stream else 0] = vid_path
#         boxes = predictor.results[i].boxes
#         if len(boxes) == 0:
#             continue
#         # 构建完整的检测结果数组
#         det = np.zeros((len(boxes), 6))  # 初始化6列的数组
#         det[:, :4] = boxes.xyxy.cpu().numpy()  # 边界框坐标
#         det[:, 4] = boxes.conf.cpu().numpy()   # 置信度
#         det[:, 5] = boxes.cls.cpu().numpy()    # 类别ID

#         # det = (predictor.results[i].obb if is_obb else predictor.results[i].boxes).cpu().numpy()
#         # if len(det) == 0:
#         #     continue
#         # 添加调试信息
#         print("Before tracking - Detection class IDs:", det[:, -1])
#         # 打印检测框的形状和内容
#         print("Detection shape:", det.shape)
#         print("Detection content:", det)

#         tracks = tracker.update(det, im0s[i])
#         if len(tracks) == 0:
#             continue
#         # 添加调试信息
#         print("After tracking - Track class IDs:", tracks[:, -1])
#         idx = tracks[:, -1].astype(int)
#         predictor.results[i] = predictor.results[i][idx]


#         # 更新结果
#         if len(tracks):
#             idx = tracks[:, -1].astype(int)
#             predictor.results[i] = predictor.results[i][idx]
            
#             update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
#             predictor.results[i].update(**update_args)

def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    path, im0s = predictor.batch[:2]
    
    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"
    
    for i in range(len(im0s)):
        tracker = predictor.trackers[i if is_stream else 0]
        
        # Reset tracker if needed
        vid_path = predictor.save_dir / Path(path[i]).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path
            
        boxes = predictor.results[i].boxes
        if len(boxes) == 0:
            continue
            
        # Prepare detection results in numpy array format
        det = boxes.cpu().numpy()
        # Convert boxes to required format [x1, y1, x2, y2, conf, cls]
        det_formatted = np.zeros((len(det), 6))
        
        # Get bounding box coordinates and ensure they are valid
        xyxy = boxes.xyxy.cpu().numpy()
        xyxy = np.nan_to_num(xyxy, nan=0.0, posinf=1e6, neginf=0.0)
        # Ensure minimum box size
        xyxy[:, 2:] = np.maximum(xyxy[:, 2:], xyxy[:, :2] + 1)
        
        det_formatted[:, :4] = xyxy
        det_formatted[:, 4] = np.nan_to_num(boxes.conf.cpu().numpy(), nan=0.0)
        det_formatted[:, 5] = 0  # All detections are of class 'drone'
        
        try:
            tracks = tracker.update(det_formatted, im0s[i])
            if len(tracks) == 0:
                continue

            # Clean up tracking results
            tracks = np.nan_to_num(tracks, nan=0.0, posinf=1e6, neginf=0.0)
            
            # Ensure minimum box dimensions
            tracks[:, 2:4] = np.maximum(tracks[:, 2:4], tracks[:, :2] + 1)
            
            # Set class ID to 0
            tracks[:, -2] = 0
            
            idx = tracks[:, -1].astype(int)
            predictor.results[i] = predictor.results[i][idx]
            
            update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
            predictor.results[i].update(**update_args)
            
        except Exception as e:
            print(f"Error during tracking: {e}")
            continue

def register_tracker(model: object, persist: bool) -> None:
    """
    Register tracking callbacks to the model for object tracking during prediction.

    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.

    Examples:
        Register tracking callbacks to a YOLO model
        >>> model = YOLOModel()
        >>> register_tracker(model, persist=True)
    """
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
    model.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=persist))
