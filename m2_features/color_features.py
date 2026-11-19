"""
MODULE 2 - PHÂN TÍCH MÀU SẮC (Chương 2, 3)
    - rgb_to_hsi(): công thức RGB->HSI tự cài bằng NumPy (Ch.3) - để đưa vào báo cáo/so sánh với OpenCV
    - compute_color_histogram(): histogram Hue chuẩn hoá (Ch.2, 3)
    - extract_color_features(): mean H/S/V + tỉ lệ màu đạt (ripe_ratio)
"""
import cv2
import numpy as np
import config


def rgb_to_hsi(bgr):
    """
    Chuyển ảnh BGR (uint8) sang HSI theo công thức Chương 3 (cài từ đầu bằng NumPy).
    Trả về: H (độ, 0..360), S (0..1), I (0..1) - mỗi cái là ma trận (h, w).
        I = (R+G+B)/3
        S = 1 - 3*min(R,G,B)/(R+G+B)
        theta = arccos( 0.5*((R-G)+(R-B)) / sqrt((R-G)^2 + (R-B)(G-B)) )
        H = theta nếu B <= G, ngược lại 360 - theta
    """
    b, g, r = [bgr[..., i].astype(np.float64) / 255.0 for i in range(3)]
    eps = 1e-8
    i = (r + g + b) / 3.0
    s = 1.0 - 3.0 * np.minimum(np.minimum(r, g), b) / (r + g + b + eps)
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + eps
    theta = np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0)))
    h = np.where(b <= g, theta, 360.0 - theta)
    return h, s, i


def compute_color_histogram(crop_bgr, mask, bins=32):
    """
    Histogram kênh Hue (thang OpenCV 0..179) chỉ trên pixel thuộc quả, chuẩn hoá tổng = 1.
        p(h_k) = n_k / N   (Ch.2)
    Trả về: np.ndarray shape (bins,)
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180]).flatten()
    total = hist.sum()
    return hist / total if total > 0 else hist


def _circular_mean_hue(h_vals):
    """Hue là góc (đỏ nằm ở 2 đầu 0 và 179) nên phải lấy trung bình vòng tròn."""
    ang = h_vals.astype(np.float64) * 2.0 * np.pi / 180.0
    m = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    return float((m * 180.0 / (2.0 * np.pi)) % 180.0)


def ripe_pixel_mask(h, s, v, fruit_cfg):
    """
    Cắt màu (Color Slicing, Ch.3): True cho pixel có Hue trong dải "đạt" của loại quả,
    đủ bão hoà (S) và đủ sáng (V). h,s,v là mảng cùng shape (thang OpenCV).
    """
    in_range = np.zeros(h.shape, bool)
    for lo, hi in fruit_cfg["ripe_hue_ranges"]:
        in_range |= (h >= lo) & (h <= hi)
    return in_range & (s >= fruit_cfg["ripe_min_s"]) & (v >= fruit_cfg["ripe_min_v"])


def extract_color_features(crop_bgr, mask, fruit="apple", fruit_cfg=None, bins=None):
    """
    Input : crop_bgr (h,w,3) uint8, mask (h,w) uint8 {0,255}, fruit = 'apple' | 'orange'
    Output: dict {mean_h, mean_s, mean_v, hist_h, ripe_ratio}
        ripe_ratio = tỉ lệ pixel của quả có Hue nằm trong dải "đạt" của loại quả đó,
                     đủ bão hoà (S) và đủ sáng (V). Dải được chỉnh trong config.FRUIT_CFG.
    """
    fruit_cfg = fruit_cfg or config.FRUIT_CFG[fruit]
    bins = bins or config.COLOR["hist_bins"]

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    inside = mask > 0
    n = int(inside.sum())
    if n == 0:
        return dict(mean_h=0.0, mean_s=0.0, mean_v=0.0,
                    hist_h=np.zeros(bins), ripe_ratio=0.0)

    h, s, v = hsv[..., 0][inside], hsv[..., 1][inside], hsv[..., 2][inside]

    ripe = ripe_pixel_mask(h, s, v, fruit_cfg)

    return dict(
        mean_h=_circular_mean_hue(h),
        mean_s=float(s.mean()),
        mean_v=float(v.mean()),
        hist_h=compute_color_histogram(crop_bgr, mask, bins),
        ripe_ratio=float(ripe.sum() / n),
    )
