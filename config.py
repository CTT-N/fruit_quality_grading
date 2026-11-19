"""
CẤU HÌNH TẬP TRUNG - mọi ngưỡng/tham số nằm ở đây.
Mỗi module chỉ đọc section của mình:
    M1 -> PREPROCESS, SEGMENT, TRACKER
    M2 -> FRUIT_CFG, DEFECT, SHAPE, COLOR
    M3 -> VIDEO, THRESHOLDS, DISPLAY
Hue theo thang OpenCV: 0..179 (1 đơn vị = 2 độ).
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")

# ---------------------------------------------------------------- M1
PREPROCESS = dict(
    median_k=5,          # Median filter (Ch.2) - khử nhiễu xung
    gaussian_k=5,        # Gaussian filter (Ch.2) - làm mịn
    gaussian_sigma=1.0,
    gamma=1.0,           # Gamma (Ch.2): <1 sáng lên, >1 tối đi; 1.0 = tắt
    equalize_v=False,    # Cân bằng lược đồ kênh V (Ch.2)
)
SEGMENT = dict(
    method="bg_dist",     # "bg_dist" (khuyên dùng) | "otsu_s" | "kmeans"
    bg_model="surface",   # "surface": nền là mặt cong mượt (chịu vignette) | "const": 1 màu hằng từ 4 góc
    bg_ring=0.08,         # surface: bề rộng dải viền dùng để khớp nền (tỉ lệ cạnh ngắn)
    work_size=480,        # tách nền ở độ phân giải <= work_size (cạnh dài) cho nhanh, mặt nạ được phóng lại về gốc
    bg_refit=True,        # surface: vòng 2 khớp lại nền bằng mọi pixel ngoài quả (tốt cho quả chiếm nhiều khung)
    bg_patch=12,          # const/kmeans: kích thước mảnh góc ảnh để ước lượng màu nền
    min_dist=30,          # bg_dist: ngưỡng = Otsu kẹp trong [min_dist, max_dist]
    max_dist=60,          #   (min chống phân đoạn nhiễu khi băng chuyền trống; max chống Otsu cắt nhầm giữa 2 vùng quả)
    s_min=50, s_max=100,  # otsu_s: Otsu kênh S kẹp trong [s_min, s_max]
    suppress_shadow=True, # loại bóng đổ: pixel cùng sắc nền nhưng tối hơn
    shadow_dS=25,
    kmeans_k=3, kmeans_bg_dist=40,   # kmeans: cụm nào có tâm cách màu nền < kmeans_bg_dist là nền
    open_k=5, close_k=9,  # Morphology (Ch.5)
    fill_holes=True,
    min_area=3000,        # bỏ vùng nhỏ hơn (pixel)
)
TRACKER = dict(max_dist=80, max_missing=5)

# ---------------------------------------------------------------- M2
# Dải Hue "đạt" để tính ripe_ratio. Chỉnh bằng tập calibration.
FRUIT_CFG = {
    "apple": dict(
        ripe_hue_ranges=[(0, 10), (165, 179), (25, 85)],  # đỏ + vàng-xanh lá (loại nâu 10..25)
        ripe_min_s=60, ripe_min_v=70,
    ),
    "orange": dict(
        ripe_hue_ranges=[(8, 24)],
        ripe_min_s=80, ripe_min_v=80,
    ),
}
COLOR = dict(hist_bins=32)

DEFECT = dict(
    smooth_k=5,             # Gaussian trước khi đo khoảng cách màu (Ch.2)
    erode_px=4,             # co mặt nạ tối thiểu (pixel) để bỏ viền quả (viền hay tối do bóng)
    erode_ratio=0.04,       # ... hoặc theo tỉ lệ với d_e (lấy giá trị lớn hơn)
    ref_sigma_ratio=0.12,   # sigma Gaussian ước lượng màu lành cục bộ = ratio * d_e
    d0_min=45.0,            # D0 (Ch.3): khoảng cách RGB tới màu lành CỤC BỘ để coi là bất thường
    use_otsu=True,          # ngưỡng = max(Otsu(D), d0_min)  (Ch.6)
    d_global_min=85.0,      # khoảng cách RGB tới màu lành TOÀN QUẢ; bắt vùng thối rộng
    kmeans_sep_min=60.0,    # 2 cụm K-means cách nhau >= giá trị này (khi không phân biệt được bằng dải màu đạt) mới coi là 2 vùng
    dark_v=45,              # pixel quá tối (V < dark_v) coi là hỏng/thối đen
    highlight_s_max=45,     # điểm chói phản quang (S thấp & V cao) KHÔNG tính là vết hỏng
    highlight_v_min=210,
    open_k=3, close_k=7,    # Morphology (Ch.5)
    min_spot_area_ratio=0.002,   # vết nhỏ hơn 0.2% diện tích quả bị bỏ
    min_spot_area_px=15,
)
SHAPE = dict(hu_log_scale=True)

# ---------------------------------------------------------------- M3
VIDEO = dict(
    fps=25, width=960, height=540,
    belt_color=None,             # None = tự lấy trung vị màu nền các ảnh; hoặc (B,G,R) cố định
    flatten_bg=True,             # bù vignette của ảnh gốc trước khi dán lên băng chuyền (giả định băng chuyền sáng đều)
    fruit_size=320,              # cạnh ảnh quả (đã letterbox ở curate_dataset)
    speed_px=10,                 # px/frame
    gap_px=180,                  # khoảng hở giữa 2 quả
    feather=28,                  # làm mờ mép ảnh khi dán lên băng chuyền (mép ảnh không lộ ra thành "khung")
    y_jitter=15,                 # lệch dọc ngẫu nhiên +-
)
PIPELINE = dict(
    zone=(0.35, 0.65),           # vùng quyết định theo chiều ngang (tỉ lệ chiều rộng khung)
    vote_frames=3,               # số lần đo/quả để bỏ phiếu đa số
    vote_step=2,                 # cách nhau 2 frame (giảm tải tính toán)
)
# Luật phân loại (Ch.0, 8). Giá trị khởi điểm theo giáo trình; calibrate.py sẽ tìm giá trị tốt hơn trên tập calib.
#   LOAI 3 : defect > g2_max_defect  HOAC  circularity < g3_max_circ  HOAC  ripe < g3_max_ripe
#   LOAI 1 : defect < g1_max_defect  VA    circularity >= g1_min_circ  VA    ripe >= g1_min_ripe
#   LOAI 2 : con lai
THRESHOLDS = dict(
    g1_max_defect=1.0, g1_min_circ=0.85, g1_min_ripe=0.80,
    g2_max_defect=10.0,
    g3_max_circ=0.70, g3_max_ripe=0.30,
)
DISPLAY = dict(font_scale=0.6, thickness=2)
