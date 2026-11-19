"""
HỢP ĐỒNG DỮ LIỆU CHUNG GIỮA 3 MODULE
=====================================
CẢ NHÓM CHỐT FILE NÀY. KHÔNG TỰ Ý SỬA SAU KHI ĐÃ CHỐT.

Luồng dữ liệu:
    M1 (segmentation)  --FruitObject-->  M2 (features)  --feature dict-->  M3 (classify/viz)
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np


@dataclass
class FruitObject:
    """Một quả đã được M1 tách khỏi nền trong 1 frame."""
    track_id: int                 # ID theo dõi qua các frame (-1 nếu chưa gán)
    frame_idx: int                # chỉ số frame chứa quả
    bbox: Tuple[int, int, int, int]       # (x, y, w, h) trên frame gốc
    centroid: Tuple[float, float]         # (cx, cy) trên frame gốc
    crop_bgr: np.ndarray          # ảnh BGR (uint8) cắt theo bbox, shape (h, w, 3)
    mask: np.ndarray              # uint8 {0, 255}, shape (h, w), cùng kích thước crop_bgr
    contour: np.ndarray           # contour OpenCV của quả (toạ độ trong HỆ CROP)


# Các key BẮT BUỘC của feature dict (M2 trả về, M3 nhận)
FEATURE_KEYS = (
    # --- màu sắc ---
    "mean_h", "mean_s", "mean_v",   # float; H theo thang OpenCV 0..179, S,V 0..255
    "hist_h",                       # np.ndarray (bins,), chuẩn hoá tổng = 1
    "ripe_ratio",                   # float 0..1 : tỉ lệ pixel quả có màu "đạt"
    # --- khuyết tật ---
    "defect_ratio",                 # float 0..100 (%)
    "n_spots",                      # int : số vết hỏng
    "defect_area_px",               # int : tổng diện tích vết hỏng (pixel)
    "defect_mask",                  # uint8 {0,255} cùng kích thước crop
    # --- hình dạng / kích thước ---
    "area_px",                      # int   : diện tích quả (số pixel)
    "perimeter_px",                 # float : chu vi
    "circularity",                  # float 0..1 : 4*pi*A / p^2
    "d_e_px",                       # float : đường kính hiệu dụng 2*sqrt(A/pi)
    "aspect_ratio",                 # float >= 1
    "hu",                           # np.ndarray (7,) : Hu moments (log-scale)
)


def validate_features(feat: Dict[str, Any]) -> None:
    """Gọi hàm này để chắc chắn feature dict đủ key. Raise KeyError nếu thiếu."""
    missing = [k for k in FEATURE_KEYS if k not in feat]
    if missing:
        raise KeyError(f"Feature dict thiếu key: {missing}")


# Nhãn phân loại (dùng ASCII vì cv2.putText không vẽ được tiếng Việt có dấu)
GRADE_1 = "LOAI 1"   # Dat - xuat khau
GRADE_2 = "LOAI 2"   # Dat - noi dia
GRADE_3 = "LOAI 3"   # Khong dat - phe pham
GRADES = (GRADE_1, GRADE_2, GRADE_3)


def is_pass(label: str) -> bool:
    """Quy ước đánh giá: Loại 1, Loại 2 = ĐẠT ; Loại 3 = KHÔNG ĐẠT."""
    return label in (GRADE_1, GRADE_2)
