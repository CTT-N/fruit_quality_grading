"""
MODULE 2 - ĐẶC TRƯNG KÍCH THƯỚC & HÌNH DẠNG (Chương 7)
    Area A = số pixel của vùng
    Perimeter p = độ dài đường biên
    Circularity = 4*pi*A / p^2   (=1 với hình tròn hoàn hảo)
    d_e = 2*sqrt(A/pi)           (đường kính hiệu dụng)
    Hu moments (7 mô men bất biến)
"""
import cv2
import numpy as np
import config


def _largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def smooth_contour_perimeter(contour, window=9):
    """
    Chu vi tính trên contour đã làm trơn (trung bình trượt vòng tròn).
    Lý do: đường biên bậc thang của pixel làm chu vi bị thổi phồng ~5% => Circularity của hình tròn
    hoàn hảo chỉ ~0.9. Làm trơn giúp Circularity ~1.0 như lý thuyết.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < window * 2:
        return float(cv2.arcLength(contour, True))
    pad = window // 2
    ext = np.vstack([pts[-pad:], pts, pts[:pad]])          # nối vòng
    ker = np.ones(window) / window
    xs = np.convolve(ext[:, 0], ker, mode="valid")
    ys = np.convolve(ext[:, 1], ker, mode="valid")
    sm = np.stack([xs, ys], axis=1)
    return float(np.sqrt(((sm - np.roll(sm, -1, axis=0)) ** 2).sum(axis=1)).sum())


def compute_hu_moments(mask, log_scale=True):
    """7 mô men bất biến Hu của mặt nạ nhị phân. log_scale=True để các giá trị cùng cỡ, dễ so sánh."""
    m = cv2.moments(mask, binaryImage=True)
    hu = cv2.HuMoments(m).flatten()
    if log_scale:
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu


def extract_shape_features(mask, cfg=None):
    """
    Input : mask uint8 {0,255} (h,w)
    Output: dict {area_px, perimeter_px, circularity, d_e_px, aspect_ratio, hu}
    Lưu ý: Circularity được kẹp trong [0,1] vì lỗi lượng tử hoá pixel có thể làm nhích > 1.
    """
    cfg = cfg or config.SHAPE
    c = _largest_contour(mask)
    if c is None or cv2.contourArea(c) <= 0:
        return dict(area_px=0, perimeter_px=0.0, circularity=0.0, d_e_px=0.0,
                    aspect_ratio=1.0, hu=np.zeros(7))

    area_px = int(cv2.countNonZero(mask))          # A = tổng số pixel (Ch.7)
    A = float(cv2.contourArea(c))                  # diện tích theo contour, dùng cho Circularity
    p = smooth_contour_perimeter(c)                # chu vi (contour đã làm trơn)
    circularity = min(1.0, 4.0 * np.pi * A / (p ** 2)) if p > 0 else 0.0
    d_e = 2.0 * np.sqrt(area_px / np.pi)

    (_, _), (w, h), _ = cv2.minAreaRect(c)
    aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0

    return dict(area_px=area_px, perimeter_px=p, circularity=float(circularity),
                d_e_px=float(d_e), aspect_ratio=float(aspect),
                hu=compute_hu_moments(mask, cfg["hu_log_scale"]))
