"""
TEST MODULE 2 (chạy ở thư mục gốc dự án):
    python -m m2_features.test_m2                       # test bằng quả giả lập
    python -m m2_features.test_m2 --image path/to.jpg --fruit apple   # test ảnh thật (mask tạm bằng Otsu-S)
Kết quả: in bảng đặc trưng + lưu ảnh minh hoạ vào outputs/figures/ (dùng cho báo cáo).
"""
import argparse
import os
import cv2
import numpy as np

import config
from common.contracts import FruitObject
from common.mock import make_mock_fruit
from common.utils import imread, imwrite, montage
from .extract_all import extract_all_features


def _overlay(obj, feat):
    """Vẽ viền quả (xanh lá) + viền vùng hỏng (vàng-cyan) + phủ nhẹ lên vùng hỏng."""
    vis = obj.crop_bgr.copy()
    dm = feat["defect_mask"]
    tint = np.zeros_like(vis); tint[..., 1:] = 255                # cyan (BGR: 255,255,0 -> ở đây dùng vàng 0,255,255)
    sel = dm > 0
    vis[sel] = (0.55 * vis[sel] + 0.45 * tint[sel]).astype(np.uint8)
    cnts, _ = cv2.findContours(dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
    cv2.drawContours(vis, [obj.contour], -1, (0, 255, 0), 2)
    return vis


def _quick_mask_object(img):
    """Mask tạm (Otsu kênh S + CC lớn nhất) - CHỈ để TV2 test ảnh thật khi M1 chưa xong."""
    img = cv2.resize(img, (300, int(300 * img.shape[0] / img.shape[1])))
    s = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1], (5, 5), 0)
    _, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    m = np.zeros_like(m); cv2.drawContours(m, [c], -1, 255, -1)
    return FruitObject(0, 0, (0, 0, img.shape[1], img.shape[0]), (0, 0), img, m, c)


def _print_row(name, f):
    print(f"{name:<22} defect={f['defect_ratio']:6.2f}%  spots={f['n_spots']:<2d} "
          f"circ={f['circularity']:.3f}  ripe={f['ripe_ratio']:.2f}  "
          f"d_e={f['d_e_px']:.0f}px  AR={f['aspect_ratio']:.2f}  "
          f"H={f['mean_h']:.0f} S={f['mean_s']:.0f} V={f['mean_v']:.0f}")


def run_mock():
    cases = [
        ("apple healthy",    dict(kind="apple_red")),
        ("apple 1 small spot", dict(kind="apple_red", n_spots=1, spot_radius=14)),
        ("apple 3 spots",    dict(kind="apple_red", n_spots=3, spot_radius=25)),
        ("apple rotten big", dict(kind="apple_red", rot_fraction=0.45, n_spots=2, spot_radius=30)),
        ("apple deformed",   dict(kind="apple_red", elongation=2.5)),
        ("green apple ok",   dict(kind="apple_green")),
        ("orange healthy",   dict(kind="orange")),
        ("orange 2 spots",   dict(kind="orange", n_spots=2, spot_radius=22)),
    ]
    tiles, res = [], {}
    for i, (name, kw) in enumerate(cases):
        obj = make_mock_fruit(seed=i, **kw)
        fruit = "orange" if kw["kind"] == "orange" else "apple"
        f = extract_all_features(obj, fruit)
        res[name] = f
        _print_row(name, f)
        tiles.append(cv2.resize(_overlay(obj, f), (220, 220)))
    # sanity check
    assert res["apple healthy"]["defect_ratio"] < 1.0, "quả lành mà báo lỗi >= 1% -> chỉnh DEFECT.d0_min"
    assert res["apple healthy"]["circularity"] > 0.85, "quả tròn mà Circularity <= 0.85 -> kiểm tra cách tính chu vi"
    assert res["apple 3 spots"]["n_spots"] == 3
    assert res["apple rotten big"]["defect_ratio"] > 30.0, "vùng thối rộng bị bỏ sót"
    assert res["apple deformed"]["circularity"] < 0.85
    print("\n>>> TẤT CẢ SANITY CHECK ĐẠT")
    grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    out = os.path.join(config.OUT_DIR, "figures", "m2_demo_mock.png")
    imwrite(out, grid); print("Đã lưu:", out)


def run_image(path, fruit):
    obj = _quick_mask_object(imread(path))
    f = extract_all_features(obj, fruit)
    _print_row(os.path.basename(path), f)
    vis = montage([obj.crop_bgr, obj.mask, _overlay(obj, f)],
                  ["goc", "mask (tam)", "khuyet tat (do)"])
    out = os.path.join(config.OUT_DIR, "figures", "m2_" + os.path.splitext(os.path.basename(path))[0] + ".png")
    imwrite(out, vis); print("Đã lưu:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image"); ap.add_argument("--fruit", default="apple", choices=list(config.FRUIT_CFG))
    a = ap.parse_args()
    run_image(a.image, a.fruit) if a.image else run_mock()
