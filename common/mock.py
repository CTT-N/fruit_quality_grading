"""
QUẢ GIẢ LẬP để các thành viên test module của mình mà KHÔNG phải chờ nhau.
    - TV2 dùng make_mock_fruit() để test features khi M1 chưa xong.
    - TV3 dùng để test classifier / overlay.
Không dùng file này trong sản phẩm cuối, chỉ để test.
"""
import cv2
import numpy as np
from .contracts import FruitObject

# BGR cơ sở
_BASE = {
    "apple_red": (40, 40, 190),
    "apple_green": (70, 170, 90),
    "orange": (30, 140, 245),
}


def make_mock_fruit(kind="apple_red", n_spots=0, spot_radius=22, rot_fraction=0.0,
                    elongation=1.0, size=300, seed=0, track_id=0, frame_idx=0,
                    bg_color=(70, 70, 70)):
    """
    Tạo 1 FruitObject giả.
        kind          : 'apple_red' | 'apple_green' | 'orange'
        n_spots       : số vết thâm nâu/đen
        spot_radius   : bán kính mỗi vết (px)
        rot_fraction  : tỉ lệ (0..1) chiều cao quả bị thối lan (0 = không)
        elongation    : >1 => quả méo (ellipse) để Circularity giảm
        bg_color      : màu nền (BGR) quanh quả trong crop
    """
    rng = np.random.default_rng(seed)
    pad = 10
    ry = int(size / 2 / np.sqrt(elongation))
    rx = int(size / 2 * np.sqrt(elongation))
    h = 2 * ry + 2 * pad
    w = 2 * rx + 2 * pad
    cy, cx = h // 2, w // 2

    # mặt nạ quả
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    # màu có shading (sáng giữa, tối rìa) + nhiễu vân
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    shade = np.clip(1.05 - 0.45 * d ** 2, 0.5, 1.05)[..., None]   # sáng giữa, tối rìa (không bão hoà)
    base = np.array(_BASE[kind], np.float32)[None, None, :]
    img = base * shade + rng.normal(0, 4, (h, w, 1))
    img = np.clip(img, 0, 255).astype(np.uint8)

    # vết thâm (nâu sẫm)
    spot_mask = np.zeros((h, w), np.uint8)
    a0 = rng.uniform(0, 2 * np.pi)
    for k in range(n_spots):                       # đặt vết cách đều nhau để không chồng lên nhau
        ang, r = a0 + 2 * np.pi * k / max(1, n_spots), 0.45
        px, py = int(cx + r * rx * np.cos(ang)), int(cy + r * ry * np.sin(ang))
        cv2.circle(spot_mask, (px, py), spot_radius, 255, -1)
    if rot_fraction > 0:
        cv2.rectangle(spot_mask, (0, cy + ry - int(2 * ry * rot_fraction)), (w, h), 255, -1)
    spot_mask = cv2.GaussianBlur(spot_mask, (9, 9), 0)
    spot_mask = cv2.bitwise_and(spot_mask, mask)
    brown = np.array((20, 45, 80), np.float32)[None, None, :]  # nâu sẫm (BGR)
    alpha = (spot_mask.astype(np.float32) / 255.0)[..., None]
    img = (img * (1 - alpha) + brown * alpha).astype(np.uint8)

    # nền quanh quả: xám băng chuyền
    bg = np.full_like(img, bg_color)
    crop = np.where(mask[..., None] > 0, img, bg)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    return FruitObject(track_id=track_id, frame_idx=frame_idx, bbox=(0, 0, w, h),
                       centroid=(cx, cy), crop_bgr=crop, mask=mask, contour=contour)
