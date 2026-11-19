"""
TEST MODULE 1 (chạy ở thư mục gốc dự án):
    python -m m1_segmentation.test_m1                                    # test bằng cảnh giả lập
    python -m m1_segmentation.test_m1 --folder data/calib/apple/healthy  # xem tách nền trên ảnh thật
    python -m m1_segmentation.test_m1 --folder data/calib/orange/rotten --method kmeans --n 12
Kết quả ảnh lưu ở outputs/figures/.
"""
import argparse
import os
import cv2
import numpy as np

import config
from common.mock import make_mock_fruit
from common.utils import imread, imwrite, montage
from .pipeline_m1 import detect_fruits
from .objects import extract_boundary, is_fully_visible
from .tracker import CentroidTracker

BG_GRAY = (232, 230, 228)        # xám ngả trắng (BGR)
BG_BLUE = (244, 232, 218)        # lam nhạt ngả trắng (BGR: B cao hơn R)


def build_scene(bg, fruit_specs, size=(960, 540), noise=3.0, shadow=True, seed=0, vignette=0.0):
    """Dựng 1 frame: nền + bóng đổ + các quả. Trả về (frame, [mask_gt_full,...])."""
    W, H = size
    rng = np.random.default_rng(seed)
    scene = np.full((H, W, 3), bg, np.float32)
    if vignette > 0:                       # nền tối dần ra góc + hơi lệch trái-phải (giống ảnh thật)
        yy, xx = np.mgrid[0:H, 0:W]
        r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
        scene *= (1 - vignette * r2 - 0.08 * xx / W)[..., None]
    scene = np.clip(scene, 0, 255).astype(np.uint8)
    gts, placed = [], []
    for kw, x0, y0 in fruit_specs:
        o = make_mock_fruit(bg_color=bg, **kw)
        placed.append((o, x0, y0))
    if shadow:                                     # bóng đổ mềm lệch xuống dưới-phải
        sh = np.zeros((H, W), np.float32)
        for o, x0, y0 in placed:
            h, w = o.mask.shape
            cv2.ellipse(sh, (x0 + w // 2 + 10, y0 + h // 2 + 18), (w // 2 - 5, h // 2 - 5), 0, 0, 360, 1.0, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), 9)
        scene = (scene * (1 - 0.14 * sh[..., None])).astype(np.uint8)
    for o, x0, y0 in placed:
        h, w = o.mask.shape
        xs, ys = max(x0, 0), max(y0, 0)
        xe, ye = min(x0 + w, W), min(y0 + h, H)
        sub_m = o.mask[ys - y0:ye - y0, xs - x0:xe - x0] > 0
        roi = scene[ys:ye, xs:xe]
        roi[sub_m] = o.crop_bgr[ys - y0:ye - y0, xs - x0:xe - x0][sub_m]
        g = np.zeros((H, W), np.uint8)
        g[ys:ye, xs:xe][sub_m] = 255
        gts.append(g)
    if noise > 0:
        scene = np.clip(scene + rng.normal(0, noise, scene.shape), 0, 255).astype(np.uint8)
    return scene, gts


def _iou(a, b):
    a, b = a > 0, b > 0
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def _full_mask(obj, shape):
    m = np.zeros(shape[:2], np.uint8)
    x, y, w, h = obj.bbox
    m[y:y + h, x:x + w] = obj.mask
    return m


def match_iou(objs, gts, shape):
    """IoU tốt nhất của mỗi mặt nạ chuẩn với các quả phát hiện được."""
    fm = [_full_mask(o, shape) for o in objs]
    return [max([_iou(f, g) for f in fm], default=0.0) for g in gts]


def build_still(size_fruit, kind, vg, bg=(215, 212, 210), seed=0, S=320, **kw):
    """Ảnh tĩnh 320x320 giống dataset thật: 1 quả ở giữa, nền xám tối dần ra góc + lệch trái-phải."""
    rng = np.random.default_rng(seed)
    o = make_mock_fruit(kind, size=size_fruit, seed=seed, bg_color=bg, **kw)
    yy, xx = np.mgrid[0:S, 0:S]
    r2 = ((xx - S / 2) / (S / 2)) ** 2 + ((yy - S / 2) / (S / 2)) ** 2
    sc = (np.full((S, S, 3), bg, np.float32) * (1 - vg * r2 - 0.10 * xx / S)[..., None]).clip(0, 255).astype(np.uint8)
    h, w = o.mask.shape
    x0, y0 = (S - w) // 2 + int(rng.integers(-20, 20)), (S - h) // 2 + int(rng.integers(-20, 20))
    xs, ys, xe, ye = max(x0, 0), max(y0, 0), min(x0 + w, S), min(y0 + h, S)
    m = o.mask[ys - y0:ye - y0, xs - x0:xe - x0] > 0
    sc[ys:ye, xs:xe][m] = o.crop_bgr[ys - y0:ye - y0, xs - x0:xe - x0][m]
    g = np.zeros((S, S), np.uint8); g[ys:ye, xs:xe][m] = 255
    return sc, g


def run_mock():
    specs = [(dict(kind="apple_red", size=230, seed=1), 40, 60),
             (dict(kind="apple_green", size=230, n_spots=2, spot_radius=18, seed=2), 330, 60),
             (dict(kind="orange", size=230, rot_fraction=0.4, seed=3), 620, 60)]
    print("== IoU trung bình theo phương pháp x nền ==")
    results = {}
    for bg_name, bg in [("xam-trang", BG_GRAY), ("lam-nhat", BG_BLUE)]:
        frame, gts = build_scene(bg, specs, seed=5)
        for method in ("bg_dist", "otsu_s", "kmeans"):
            cfg = dict(config.SEGMENT, method=method)
            objs, mask = detect_fruits(frame, 0, seg_cfg=cfg)
            ious = match_iou(objs, gts, frame.shape)
            results[(bg_name, method)] = (len(objs), float(np.mean(ious)), frame, objs, mask)
            print(f"{bg_name:<10} {method:<8} so_qua={len(objs)}  IoU={np.mean(ious):.3f}  "
                  f"(min={min(ious):.3f})")

    # --- kiểm tra bắt buộc cho phương pháp chính ---
    for bg_name in ("xam-trang", "lam-nhat"):
        n, iou, *_ = results[(bg_name, "bg_dist")]
        assert n == 3, f"[{bg_name}] phát hiện {n} quả, cần 3"
        assert iou > 0.95, f"[{bg_name}] IoU {iou:.3f} < 0.95"

    # băng chuyền trống (chỉ nền + nhiễu + bóng) -> 0 quả
    for bg in (BG_GRAY, BG_BLUE):
        empty, _ = build_scene(bg, [], noise=4.0, seed=9)
        objs, _ = detect_fruits(empty)
        assert len(objs) == 0, "băng chuyền trống mà phát hiện quả (tăng min_dist/min_area)"

    # quả bị cắt bởi mép khung
    frame, _ = build_scene(BG_GRAY, [(dict(kind="apple_red", size=230, seed=1), 800, 100)])
    objs, _ = detect_fruits(frame)
    assert len(objs) == 1 and not is_fully_visible(objs[0], frame.shape), "quả cắt mép phải bị đánh dấu"

    # tracking: 3 quả trượt 6 px/frame trong 12 frame -> ID ổn định
    tr, ids_seen = CentroidTracker(), []
    for t in range(12):
        sp = [(kw, x + 6 * t, y) for kw, x, y in specs]
        frame, _ = build_scene(BG_BLUE, sp, seed=t)
        objs, _ = detect_fruits(frame, t)
        ids_seen.append(tuple(o.track_id for o in tr.update(objs)))
    assert len(set(ids_seen)) == 1 and ids_seen[0] == (0, 1, 2), f"ID không ổn định: {set(ids_seen)}"

    # nền vignette (tối dần ra góc) - tình huống của ảnh thật
    print("\n== Nền vignette ==")
    for vg in (0.12, 0.2):
        frame, gts = build_scene(BG_GRAY, specs, seed=5, vignette=vg)
        for model in ("const", "surface"):
            cfg = dict(config.SEGMENT, bg_model=model)
            objs, mask = detect_fruits(frame, 0, seg_cfg=cfg)
            ious = match_iou(objs, gts, frame.shape)
            print(f"vignette={vg:<5} bg_model={model:<8} so_vat={len(objs)} IoU={np.mean(ious):.3f}")
            if model == "surface":
                assert len(objs) == 3 and np.mean(ious) > 0.95, f"nền vignette {vg}: IoU {np.mean(ious):.3f}"

    # ảnh tĩnh giống dataset thật: quả chiếm ~25-45% khung, nền vignette mạnh
    print("\n== Ảnh tĩnh 320x320, nền vignette ==")
    cases = [("apple_red", 150, {}), ("apple_green", 170, {}), ("orange", 180, dict(rot_fraction=0.3)),
             ("apple_red", 160, dict(n_spots=3, spot_radius=18)), ("orange", 130, {}),
             ("apple_green", 190, dict(rot_fraction=0.45))]
    for vg in (0.0, 0.15, 0.3):
        ious, nobj = [], []
        for i, (k, sz, kw) in enumerate(cases):
            sc, g = build_still(sz, k, vg, seed=i, **kw)
            objs, _ = detect_fruits(sc)
            nobj.append(len(objs)); ious.append(max(match_iou(objs, [g], sc.shape)) if objs else 0.0)
        print(f"vignette={vg:<5} IoU_tb={np.mean(ious):.3f} (min={min(ious):.3f})  dung 1 vat: {sum(n == 1 for n in nobj)}/{len(nobj)}")
        assert min(ious) > 0.93 and all(n == 1 for n in nobj), f"ảnh tĩnh vignette {vg} tách sai"

    print("\n>>> TẤT CẢ SANITY CHECK ĐẠT")
    # ảnh minh hoạ
    rows = []
    for bg_name in ("xam-trang", "lam-nhat"):
        _, _, frame, objs, mask = results[(bg_name, "bg_dist")]
        vis = frame.copy()
        for o in objs:
            x, y, w, h = o.bbox
            b = extract_boundary(o.mask)
            vis[y:y + h, x:x + w][b > 0] = (0, 255, 0)
        rows.append(montage([frame, mask, vis], [bg_name + " goc", "mask", "bien (xanh)"], height=220))
    out = os.path.join(config.OUT_DIR, "figures", "m1_demo_mock.png")
    imwrite(out, np.vstack(rows)); print("Đã lưu:", out)


def run_folder(folder, n, method):
    cfg = dict(config.SEGMENT, method=method)
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))[:n]
    tiles, warn = [], 0
    for f in files:
        img = imread(os.path.join(folder, f))
        objs, mask = detect_fruits(img, 0, seg_cfg=cfg)
        ratio = float((mask > 0).mean())
        flag = ""
        if len(objs) != 1 or not (0.08 < ratio < 0.90):
            flag = "  <-- KIEM TRA"; warn += 1
        print(f"{f:<36} so_vat={len(objs)}  dien_tich_mask={ratio:5.1%}{flag}")
        vis = img.copy()
        vis[extract_boundary(mask) > 0] = (0, 255, 0)
        tiles.append(cv2.resize(np.hstack([img, vis]), (320, 160)))
    cols = 2
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out = os.path.join(config.OUT_DIR, "figures", f"m1_{os.path.basename(folder.rstrip('/\\\\'))}_{method}.png")
    imwrite(out, grid); print(f"\n{warn}/{len(files)} ảnh cần kiểm tra. Đã lưu: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder"); ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--method", default=config.SEGMENT["method"], choices=["bg_dist", "otsu_s", "kmeans"])
    a = ap.parse_args()
    run_folder(a.folder, a.n, a.method) if a.folder else run_mock()
