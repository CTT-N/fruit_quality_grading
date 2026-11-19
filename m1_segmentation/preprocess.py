"""
MODULE 1 - TIỀN XỬ LÝ (Chương 2)
    Median filter   : khử nhiễu xung (muối tiêu), giữ biên            g(x,y) = median{f(s,t)}
    Gaussian filter : làm mịn, giảm nhiễu hạt                          w(s,t) = K*exp(-(s^2+t^2)/(2*sigma^2))
    Gamma           : s = c*r^gamma, chỉnh ảnh thiếu/thừa sáng
    Histogram Eq.   : s_k = round((L-1) * sum_{j<=k} p(r_j)) trên kênh V
Lưu ý: ảnh tiền xử lý chỉ dùng cho TÁCH NỀN. Crop gửi cho M2 lấy từ frame GỐC để không làm sai màu.
"""
import cv2
import numpy as np
import config


def apply_gamma(img, gamma):
    """Biến đổi hàm mũ (Ch.2) bằng bảng tra LUT. gamma<1 sáng lên, gamma>1 tối đi, 1.0 = giữ nguyên."""
    if abs(gamma - 1.0) < 1e-6:
        return img
    lut = np.clip(255.0 * (np.arange(256) / 255.0) ** gamma, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def equalize_value_channel(img):
    """Cân bằng lược đồ (Ch.2) chỉ trên kênh V của HSV để không làm lệch màu."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def preprocess_frame(frame_bgr, cfg=None):
    """
    Input : frame BGR uint8 (h,w,3)
    Output: frame BGR uint8 cùng kích thước, đã khử nhiễu + chỉnh sáng
    """
    cfg = cfg or config.PREPROCESS
    out = cv2.medianBlur(frame_bgr, cfg["median_k"] | 1)
    k = cfg["gaussian_k"] | 1
    out = cv2.GaussianBlur(out, (k, k), cfg["gaussian_sigma"])
    out = apply_gamma(out, cfg["gamma"])
    if cfg["equalize_v"]:
        out = equalize_value_channel(out)
    return out
