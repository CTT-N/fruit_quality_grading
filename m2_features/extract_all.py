"""
MODULE 2 - HÀM DUY NHẤT MÀ M3 GỌI:  extract_all_features(obj, fruit)
"""
import config
from common.contracts import FruitObject, FEATURE_KEYS, validate_features
from .color_features import extract_color_features
from .defect_detection import detect_defects
from .shape_features import extract_shape_features


def extract_all_features(obj: FruitObject, fruit: str = "apple", cfg=None) -> dict:
    """
    Input : FruitObject (từ M1), fruit = 'apple' | 'orange'
    Output: feature dict đủ các key trong common.contracts.FEATURE_KEYS
    """
    color = extract_color_features(obj.crop_bgr, obj.mask, fruit)
    defect = detect_defects(obj.crop_bgr, obj.mask, cfg or config.DEFECT, config.FRUIT_CFG[fruit])
    shape = extract_shape_features(obj.mask)

    feat = {}
    feat.update(color)
    feat.update(defect)
    feat.update(shape)
    feat = {k: feat[k] for k in FEATURE_KEYS}   # chỉ giữ key theo hợp đồng
    validate_features(feat)
    return feat
