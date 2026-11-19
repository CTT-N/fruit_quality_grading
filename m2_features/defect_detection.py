"""
MODULE 2 - PHÁT HIỆN KHUYẾT TẬT (Chương 2, 3, 5, 6, 7)

Các bước:
  1. Làm mịn Gaussian nhẹ                                                                   (Ch.2)
  2. Co mặt nạ quả để bỏ viền (viền hay tối do bóng)                                        (Ch.5 Erosion)
  3. Xác định MÀU LÀNH tham chiếu:
       K-means K=2 trên màu pixel (Ch.6). Nếu 2 cụm tách xa nhau (có vùng hỏng rộng) -> chọn cụm
       "lành" = cụm có nhiều pixel nằm trong dải màu đạt của loại quả (Color Slicing, Ch.3);
       nếu không thì dùng trung vị toàn quả.
  4. Màu lành CỤC BỘ tại từng pixel = trung bình Gaussian có trọng số (loại pixel nghi hỏng, lặp 3 vòng)
     => bóng đổ mượt (rìa tối hơn tâm) KHÔNG bị nhận nhầm là vết; vết thâm sắc nét thì nổi lên.
  5. Khoảng cách Euclid RGB:  D(z,a) = sqrt((zR-aR)^2+(zG-aG)^2+(zB-aB)^2) > D0                (Ch.3)
     Ngưỡng cục bộ = max(Otsu(D), D0_min)                                                     (Ch.6 Otsu)
     + pixel lệch xa màu lành toàn quả (d_global_min) + pixel quá tối (V < dark_v).
     Điểm chói phản quang (S thấp, V cao) bị loại trừ.
  6. Opening xoá đốm li ti, Closing lấp lỗ                                                    (Ch.5)
  7. Connected Component Labeling: đếm vết + diện tích từng vết                               (Ch.5)
  8. P_loi = sum(A_benh) / A_qua * 100%                                                       (Ch.7)
"""
import cv2
import numpy as np
import config
from .color_features import ripe_pixel_mask


def _ellipse(k):
    k = max(1, int(k)) | 1  # ép về số lẻ
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _ripe_fraction(pix_bgr, fruit_cfg):
    """Tỉ lệ pixel (N,3 BGR uint8) nằm trong dải màu đạt."""
    hsv = cv2.cvtColor(pix_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return float(ripe_pixel_mask(hsv[:, 0], hsv[:, 1], hsv[:, 2], fruit_cfg).mean())


def estimate_healthy_reference(pix_bgr, fruit_cfg, sep_min, max_samples=4000):
    """
    Ước lượng màu "lành" của cả quả bằng K-means K=2 (Ch.6) + Color Slicing theo loại quả (Ch.3).
    Trả về (ref_healthy_bgr, split, defect_center_bgr | None)
        split=True  : phát hiện quả có 2 vùng màu khác biệt (vd vỏ đỏ + vùng thối nâu)
        split=False : chỉ 1 vùng màu (khác nhau chỉ do bóng đổ) -> ref = trung vị toàn quả
    Quy tắc tách:
        - cụm nhỏ nhất < 5% mẫu                      -> không tách
        - cả 2 cụm đều >= 80% pixel màu đạt          -> không tách (vd táo 2 màu đỏ-xanh)
        - chênh lệch tỉ lệ màu đạt giữa 2 cụm > 0.3  -> tách; cụm "lành" = cụm có nhiều màu đạt hơn
        - ngược lại: tách nếu 2 tâm cụm cách nhau >= sep_min; cụm lành = cụm lớn hơn
    """
    pix = pix_bgr.astype(np.float32)
    med = np.median(pix, axis=0)
    if len(pix) < 200:
        return med, False, None

    rng = np.random.default_rng(0)
    sub = pix[rng.choice(len(pix), min(max_samples, len(pix)), replace=False)]
    cv2.setRNGSeed(0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, cen = cv2.kmeans(sub, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    sizes = [int((lab == k).sum()) for k in (0, 1)]
    if min(sizes) < 0.05 * len(sub):
        return med, False, None

    members = [sub[lab == k] for k in (0, 1)]
    if fruit_cfg is not None:
        rf = [_ripe_fraction(m, fruit_cfg) for m in members]
        if min(rf) >= 0.8:
            return med, False, None
        if abs(rf[0] - rf[1]) > 0.3:
            h = int(np.argmax(rf))
        elif np.linalg.norm(cen[0] - cen[1]) >= sep_min:
            h = int(np.argmax(sizes))
        else:
            return med, False, None
    else:
        if np.linalg.norm(cen[0] - cen[1]) < sep_min:
            return med, False, None
        h = int(np.argmax(sizes))
    return np.median(members[h], axis=0), True, np.median(members[1 - h], axis=0)


def estimate_local_reference(img_f32, inside, sigma, init_w, d_thr, fallback, n_iter=3):
    """
    Màu lành cục bộ (h,w,3) bằng normalized Gaussian convolution trong vùng quả.
    init_w: trọng số khởi tạo (loại sẵn vùng hỏng rộng). Mỗi vòng loại thêm pixel có D > d_thr.
    """
    w = init_w.astype(np.float32)
    ref = np.tile(fallback.astype(np.float32), (*inside.shape, 1))
    for it in range(n_iter):
        num = cv2.GaussianBlur(img_f32 * w[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(w, (0, 0), sigma)
        ok = den > 1e-3
        ref = np.where(ok[..., None], num / np.maximum(den, 1e-3)[..., None], fallback[None, None, :])
        if it < n_iter - 1:
            d = np.sqrt(((img_f32 - ref) ** 2).sum(axis=2))
            w_new = (inside & (d < d_thr)).astype(np.float32)
            if w_new.sum() < 50:
                break
            w = w_new
    return ref.astype(np.float32)


def detect_defects(crop_bgr, mask, cfg=None, fruit_cfg=None):
    """
    Input : crop_bgr (h,w,3) uint8; mask (h,w) uint8 {0,255}
            fruit_cfg: config.FRUIT_CFG['apple'] (tuỳ chọn; giúp chọn đúng màu lành khi có vùng thối rộng)
    Output: dict
        defect_mask    uint8 {0,255} (h,w)
        defect_ratio   float, % diện tích quả bị lỗi (0..100)
        n_spots        int, số vết
        defect_area_px int, tổng diện tích vết (pixel)
        spots          list[dict(area, centroid, bbox)]
        ref_bgr        màu lành tham chiếu (để vẽ minh hoạ báo cáo)
        threshold      ngưỡng khoảng cách cục bộ đã dùng
        dist_map       bản đồ khoảng cách (để vẽ minh hoạ)
    """
    cfg = cfg or config.DEFECT
    fruit_area = int(cv2.countNonZero(mask))
    empty = dict(defect_mask=np.zeros_like(mask), defect_ratio=0.0, n_spots=0,
                 defect_area_px=0, spots=[], ref_bgr=(0, 0, 0), threshold=0.0,
                 dist_map=np.zeros(mask.shape, np.float32))
    if fruit_area == 0:
        return empty

    # 1) làm mịn nhẹ
    k = cfg["smooth_k"] | 1
    smooth = cv2.GaussianBlur(crop_bgr, (k, k), 0)
    smooth_f = smooth.astype(np.float32)

    # 2) co mặt nạ, bỏ viền
    d_e = 2.0 * np.sqrt(fruit_area / np.pi)
    e = max(int(cfg["erode_px"]), int(round(cfg["erode_ratio"] * d_e)))
    inner = cv2.erode(mask, _ellipse(2 * e + 1)) if e > 0 else mask.copy()
    if cv2.countNonZero(inner) < 50:
        inner = mask.copy()
    inside = inner > 0

    # 3) màu lành toàn quả
    ref, split, c_def = estimate_healthy_reference(smooth[inside], fruit_cfg, sep_min=cfg["kmeans_sep_min"])
    dist_g = np.sqrt(((smooth_f - ref[None, None, :]) ** 2).sum(axis=2))
    cluster_def = np.zeros(mask.shape, bool)
    if split:                                   # gán mỗi pixel vào cụm gần hơn (K-means, Ch.6)
        d_def = np.sqrt(((smooth_f - c_def[None, None, :]) ** 2).sum(axis=2))
        cluster_def = (d_def < dist_g) & inside

    # 4) màu lành cục bộ (bù bóng đổ). Nếu đã tách cụm: khởi tạo chỉ bằng pixel gần màu lành.
    sigma = max(5.0, cfg["ref_sigma_ratio"] * d_e)
    init_w = (inside & ~cluster_def) if split else inside
    if init_w.sum() < 50:
        init_w = inside
    ref_map = estimate_local_reference(smooth_f, inside, sigma, init_w, cfg["d0_min"], ref)

    # 5) khoảng cách màu + ngưỡng
    dist = np.sqrt(((smooth_f - ref_map) ** 2).sum(axis=2))
    thr = float(cfg["d0_min"])
    if cfg["use_otsu"]:
        d8 = np.clip(dist[inside], 0, 255).astype(np.uint8).reshape(-1, 1)
        otsu_t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(thr, float(otsu_t))
    cand = (dist > thr) & inside
    cand |= (dist_g > cfg["d_global_min"]) & inside          # lệch xa màu lành toàn quả
    cand |= cluster_def                                       # thuộc cụm "hỏng" (nếu đã tách cụm)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    cand |= (v < cfg["dark_v"]) & inside                      # quá tối
    highlight = (hsv[..., 1] < cfg["highlight_s_max"]) & (v > cfg["highlight_v_min"])
    cand &= ~highlight                                        # bỏ điểm chói
    raw = cand.astype(np.uint8) * 255

    # 6) Morphology: Opening (bỏ đốm nhiễu) rồi Closing (lấp lỗ)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, _ellipse(cfg["open_k"]))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, _ellipse(cfg["close_k"]))
    raw = cv2.bitwise_and(raw, mask)

    # 7) Connected Component Labeling
    n_lab, labels, stats, cents = cv2.connectedComponentsWithStats(raw, connectivity=8)
    min_area = max(cfg["min_spot_area_px"], cfg["min_spot_area_ratio"] * fruit_area)
    defect_mask = np.zeros_like(mask)
    spots = []
    for i in range(1, n_lab):  # 0 là nền
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        defect_mask[labels == i] = 255
        spots.append(dict(area=a,
                          centroid=(float(cents[i][0]), float(cents[i][1])),
                          bbox=(int(stats[i, 0]), int(stats[i, 1]),
                                int(stats[i, 2]), int(stats[i, 3]))))

    # 8) tỉ lệ lỗi
    defect_area = int(sum(s["area"] for s in spots))
    return dict(defect_mask=defect_mask,
                defect_ratio=100.0 * defect_area / fruit_area,
                n_spots=len(spots), defect_area_px=defect_area, spots=spots,
                ref_bgr=tuple(float(x) for x in ref), threshold=thr, dist_map=dist)
