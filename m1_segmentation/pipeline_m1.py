"""
MODULE 1 - HÀM GỘP MÀ M3 GỌI:  detect_fruits(frame)  rồi  tracker.update(objs)
"""
import config
from common.contracts import FruitObject
from .preprocess import preprocess_frame
from .segment import segment_fruits
from .objects import extract_objects


def detect_fruits(frame_bgr, frame_idx=0, pre_cfg=None, seg_cfg=None):
    """
    Input : frame BGR uint8 gốc (từ video hoặc ảnh)
    Output: (objects: List[FruitObject], mask_full: uint8 {0,255})
    """
    seg_cfg = seg_cfg or config.SEGMENT
    pre = preprocess_frame(frame_bgr, pre_cfg)
    mask = segment_fruits(pre, seg_cfg)
    objs = extract_objects(frame_bgr, mask, frame_idx, seg_cfg["min_area"])
    return objs, mask
