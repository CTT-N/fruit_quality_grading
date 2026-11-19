"""
MODULE 1 - TÁCH QUẢ KHỎI NỀN (Chương 3, 5, 6)

Phương pháp chính "bg_dist" (nền đơn sắc xám-trắng / lam nhạt, KỂ CẢ nền bị tối dần ra góc - vignette):
  1. Ước lượng NỀN dạng mặt cong mượt: khớp đa thức bậc cao (1,x,y,x²,xy,y²,r⁴) theo bình phương tối thiểu
     trên dải viền ảnh, lặp loại pixel ngoại lai (quả chạm mép)       -> thích ứng từng ảnh, chịu được vignette
  2. D(z) = khoảng cách Euclid RGB tới màu nền TẠI CHÍNH VỊ TRÍ ĐÓ  (Color Slicing, Ch.3)
  3. Ngưỡng = Otsu(D) kẹp trong [min_dist, max_dist]                   (Ch.6 Otsu)
  4. Loại bóng đổ: pixel cùng sắc với nền (S thấp) nhưng tối hơn nền
  5. Opening, Closing, lấp lỗ                                          (Ch.5 Morphology)
Phương pháp dự phòng: "otsu_s" (Otsu kênh S, Ch.3+6) và "kmeans" (K-means K=3, Ch.6).
"""
import cv2
import numpy as np
import config


def _ellipse(k):
    k = max(1, int(k)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def estimate_background_color(img, patch=12):
    """
    Màu nền (3,) = trung vị của trung vị 4 mảnh góc. Quả hiếm khi phủ cả 4 góc nên khá bền
    (kể cả khi quả chạm 1-2 mép ảnh).
    """
    h, w = img.shape[:2]
    p = max(1, min(patch, h // 4, w // 4))
    corners = [img[:p, :p], img[:p, -p:], img[-p:, :p], img[-p:, -p:]]
    meds = np.array([np.median(c.reshape(-1, 3), axis=0) for c in corners])
    return np.median(meds, axis=0)


def _poly_design(xn, yn):
    """Ma trận thiết kế đa thức theo toạ độ chuẩn hoá [-1,1]: [1, x, y, x^2, xy, y^2, (x^2+y^2)^2]."""
    return np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn, (xn * xn + yn * yn) ** 2], axis=-1)


def estimate_background_surface(img, ring_frac=0.08, n_iter=3, exclude=None):
    """
    Nền dạng mặt cong (h,w,3) float32. Khớp đa thức trên dải viền ảnh (rộng ring_frac*cạnh ngắn);
    mỗi vòng loại các pixel có sai số lớn (quả chạm mép ảnh, vật lạ). Tính trên ảnh thu nhỏ rồi phóng lại -> nhanh.
    exclude: mặt nạ bool/uint8 (h,w) của vùng ĐÃ BIẾT là quả. Nếu có, khớp trên MỌI pixel ngoài vùng này
             (ràng buộc nền sát quả tốt hơn so với chỉ dùng dải viền).
    """
    H, W = img.shape[:2]
    s = max(1, int(round(max(H, W) / 160)))
    small = cv2.resize(img, (max(8, W // s), max(8, H // s)), interpolation=cv2.INTER_AREA).astype(np.float32)
    h, w = small.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    A = _poly_design((xx / (w - 1)) * 2 - 1, (yy / (h - 1)) * 2 - 1).reshape(-1, 7).astype(np.float64)
    Y = small.reshape(-1, 3).astype(np.float64)
    ring_px = max(2, int(round(ring_frac * min(h, w))))
    ring = (np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)) < ring_px).ravel()

    fit_set = ring
    if exclude is not None:
        ex = cv2.resize((exclude > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).ravel() > 0
        if (~ex).sum() >= 0.12 * h * w:             # đủ pixel nền mới dùng; nếu không giữ dải viền
            fit_set = ~ex
    inl, coef = fit_set.copy(), None
    for _ in range(n_iter):
        if inl.sum() < 30:
            break
        coef, *_ = np.linalg.lstsq(A[inl], Y[inl], rcond=None)
        r = np.linalg.norm(A @ coef - Y, axis=1)
        inl = fit_set & (r < max(12.0, 3.0 * np.median(r[fit_set]) + 6.0))
    if coef is None:                                              # dự phòng: nền hằng số
        const = np.median(Y[ring], axis=0)
        return np.tile(const.astype(np.float32), (H, W, 1))
    surf = A @ coef
    lo = np.percentile(Y[inl], 1, axis=0) - 15.0            # chặn ngoại suy phi lý: nền không thể nằm ngoài
    hi = np.percentile(Y[inl], 99, axis=0) + 15.0           # khoảng giá trị thật của các pixel nền đã khớp
    surf = np.clip(surf, lo, hi).reshape(h, w, 3).astype(np.float32)
    return np.clip(cv2.resize(surf, (W, H), interpolation=cv2.INTER_LINEAR), 0, 255)


def estimate_background(img, cfg, exclude=None):
    """Nền (h,w,3) float32. bg_model='surface' (mặc định, chịu vignette) hoặc 'const' (màu hằng từ 4 góc)."""
    if cfg.get("bg_model", "surface") == "const":
        return np.tile(estimate_background_color(img, cfg["bg_patch"]).astype(np.float32), (*img.shape[:2], 1))
    return estimate_background_surface(img, cfg.get("bg_ring", 0.08), exclude=exclude)


def _bg_dist_mask(img, cfg):
    """Tính ở độ phân giải làm việc <= work_size (cạnh dài) cho nhanh; ảnh nhỏ hơn giữ nguyên."""
    H, W = img.shape[:2]
    k = max(1, int(np.ceil(max(H, W) / cfg.get("work_size", 480))))
    if k == 1:
        return _bg_dist_mask_core(img, cfg)
    small = cv2.resize(img, (W // k, H // k), interpolation=cv2.INTER_AREA)
    m = _bg_dist_mask_core(small, cfg)
    return (cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8) * 255


def _bg_dist_mask_core(img, cfg):
    f32 = img.astype(np.float32)

    def dist_mask(bg):
        dist = np.sqrt(((f32 - bg) ** 2).sum(axis=2))
        t, _ = cv2.threshold(np.clip(dist, 0, 255).astype(np.uint8), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return dist > float(np.clip(t, cfg["min_dist"], cfg["max_dist"]))

    bg = estimate_background(img, cfg)                       # vòng 1: khớp nền từ dải viền
    mask = dist_mask(bg)
    if cfg.get("bg_model", "surface") == "surface" and cfg.get("bg_refit", True) and mask.any():
        r = max(8, int(0.05 * min(img.shape[:2])))           # vòng 2: khớp lại bằng mọi pixel ngoài quả
        excl = cv2.dilate(mask.astype(np.uint8) * 255, _ellipse(2 * r + 1))
        bg = estimate_background(img, cfg, exclude=excl)
        mask = dist_mask(bg)
    if cfg["suppress_shadow"]:                               # bóng đổ: cùng sắc với nền (S thấp), tối hơn nền
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bg_hsv = cv2.cvtColor(np.clip(bg, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        shadow = ((hsv[..., 1].astype(np.int16) < bg_hsv[..., 1].astype(np.int16) + cfg["shadow_dS"])
                  & (hsv[..., 2] < bg_hsv[..., 2]) & (hsv[..., 2] > 0.4 * bg_hsv[..., 2]))
        mask &= ~shadow
    return mask.astype(np.uint8) * 255


def _otsu_s_mask(img, cfg):
    s = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    t, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(t, cfg["s_min"], cfg["s_max"]))
    return (s > thr).astype(np.uint8) * 255


def _kmeans_mask(img, cfg):
    bg = estimate_background_color(img, cfg["bg_patch"])
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    data = small.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(data, cfg["kmeans_k"], None, crit, 2, cv2.KMEANS_PP_CENTERS)
    is_bg = np.linalg.norm(centers - bg[None, :], axis=1) < cfg["kmeans_bg_dist"]
    if not is_bg.any():                                   # luôn coi cụm gần nền nhất là nền
        is_bg[np.argmin(np.linalg.norm(centers - bg[None, :], axis=1))] = True
    f = img.reshape(-1, 3).astype(np.float32)
    d = ((f[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    lab = np.argmin(d, axis=1).reshape(img.shape[:2])
    return (~is_bg[lab]).astype(np.uint8) * 255


def clean_mask(mask, open_k=5, close_k=9, fill_holes=True):
    """Opening xoá đốm nhiễu, Closing hàn khe, lấp lỗ bên trong quả (Ch.5)."""
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(open_k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(close_k))
    if fill_holes:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        m = np.zeros_like(m)
        cv2.drawContours(m, cnts, -1, 255, -1)
    return m


def segment_fruits(frame_pre, cfg=None):
    """
    Input : frame BGR đã tiền xử lý (h,w,3) uint8
    Output: mask_full uint8 {0,255} (h,w); 255 = quả
    """
    cfg = cfg or config.SEGMENT
    method = cfg["method"]
    if method == "bg_dist":
        raw = _bg_dist_mask(frame_pre, cfg)
    elif method == "otsu_s":
        raw = _otsu_s_mask(frame_pre, cfg)
    elif method == "kmeans":
        raw = _kmeans_mask(frame_pre, cfg)
    else:
        raise ValueError(f"SEGMENT.method không hợp lệ: {method}")
    return clean_mask(raw, cfg["open_k"], cfg["close_k"], cfg["fill_holes"])
