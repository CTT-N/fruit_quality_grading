"""MODULE 3 - VẼ KẾT QUẢ LÊN VIDEO (Chương 5: biên hình thái; hiển thị trực quan)"""
import cv2
import numpy as np

import config
from common.contracts import GRADE_1, GRADE_2, GRADE_3
from m1_segmentation.objects import extract_boundary

COLORS = {GRADE_1: (0, 200, 0), GRADE_2: (0, 190, 255), GRADE_3: (0, 0, 230), None: (150, 150, 150)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, txt, org, color, scale, thick, bg=True):
    (w, h), base = cv2.getTextSize(txt, FONT, scale, thick)
    x, y = org
    if bg:
        cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + base), (30, 30, 30), -1)
    cv2.putText(img, txt, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_overlay(frame, objs, infos, counts, fps, frame_idx, zone):
    """
    Input : frame BGR; objs: List[FruitObject]; infos: dict track_id -> dict(label|None, defect_ratio, circularity, defect_mask|None)
            counts: dict grade -> số quả đã chốt; fps: float; zone: (x_lo, x_hi) pixel
    Output: frame BGR đã vẽ (bản sao)
    """
    d = config.DISPLAY
    out = frame.copy()
    H, W = out.shape[:2]

    for x in zone:                                                   # vùng quyết định
        for y in range(0, H, 24):
            cv2.line(out, (int(x), y), (int(x), min(y + 12, H)), (200, 120, 0), 1)

    for o in objs:
        info = infos.get(o.track_id, {})
        label = info.get("label")
        col = COLORS.get(label, COLORS[None])
        x, y, w, h = o.bbox
        roi = out[y:y + h, x:x + w]
        b = cv2.dilate(extract_boundary(o.mask), np.ones((3, 3), np.uint8))
        roi[b > 0] = col                                             # viền quả (biên hình thái)
        dm = info.get("defect_mask")
        if dm is not None and dm.shape == o.mask.shape:               # viền vùng hỏng
            cnts, _ = cv2.findContours(dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(roi, cnts, -1, (255, 255, 0), 2)
        ty = y - 24 if y > 50 else y + h + 18
        _text(out, f"#{o.track_id} {label if label else 'DANG PHAN TICH'}", (x, ty), col, d["font_scale"], d["thickness"])
        if label:
            _text(out, f"loi {info['defect_ratio']:.1f}%  tron {info['circularity']:.2f}", (x, ty + 20),
                  (255, 255, 255), d["font_scale"] - 0.1, 1)

    n_pass = counts[GRADE_1] + counts[GRADE_2]
    n_fail = counts[GRADE_3]
    lines = [f"LOAI 1: {counts[GRADE_1]}   LOAI 2: {counts[GRADE_2]}   LOAI 3: {counts[GRADE_3]}",
             f"DAT: {n_pass}   KHONG DAT: {n_fail}   |  frame {frame_idx}  {fps:.0f} FPS"]
    cv2.rectangle(out, (0, 0), (W, 56), (30, 30, 30), -1)
    for i, s in enumerate(lines):
        cv2.putText(out, s, (10, 22 + 24 * i), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out
