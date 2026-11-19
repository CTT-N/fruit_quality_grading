"""
MODULE 3 - TẠO VIDEO BĂNG CHUYỀN MÔ PHỎNG TỪ ẢNH (Chương 1, 4)

Các quả (ảnh đã letterbox ở data/test/...) trượt ngang từ trái sang phải trên băng chuyền sáng.
Mép ảnh được làm mờ dần (feather) để không lộ khung chữ nhật. Ghi kèm ground_truth.csv.
GIẢ ĐỊNH MÔ PHỎNG (ghi vào báo cáo): băng chuyền có màu gần giống nền ảnh; quả cách đều nhau, tốc độ không đổi.

Chạy ở thư mục gốc:
    python -m m3_classify_viz.make_video --fruit apple  --split test --n 20 --out data/videos/apple_test.mp4
    python -m m3_classify_viz.make_video --fruit orange --split test --n 20 --out data/videos/orange_test.mp4
"""
import argparse
import csv
import os
import random

import cv2
import numpy as np

import config
from common.utils import imread
from m1_segmentation.curate_dataset import letterbox
from m1_segmentation.segment import estimate_background_surface


def feathered_alpha(size, feather):
    """Ma trận alpha (size,size) =1 ở giữa, giảm tuyến tính về ~0 ở mép."""
    ramp = np.minimum(np.arange(size) + 1, np.arange(size)[::-1] + 1) / float(feather)
    ramp = np.clip(ramp, 0, 1).astype(np.float32)
    return np.outer(ramp, ramp)


def flatten_background(img_bgr, ring_frac=0.08):
    """
    Bù vignette (flat-field): chia ảnh cho nền mặt cong rồi nhân với màu nền ở TÂM ảnh.
    Kết quả: nền đều màu (bằng màu nền ở tâm), vùng quả (nằm gần tâm) gần như không đổi.
    Trả về (ảnh float32, màu nền tâm (3,)).
    """
    bg = estimate_background_surface(img_bgr, ring_frac)
    h, w = bg.shape[:2]
    c = bg[int(.35 * h):int(.65 * h), int(.35 * w):int(.65 * w)].reshape(-1, 3).mean(axis=0)
    gain = np.clip(c[None, None, :] / np.maximum(bg, 1.0), 0.7, 2.0)
    return np.clip(img_bgr.astype(np.float32) * gain, 0, 255), c


def select_from_manifest(manifest_csv, data_dir, fruit, split, n_per_label, seed=0):
    """Lấy n_per_label ảnh 'healthy' + n_per_label ảnh 'rotten', trộn ngẫu nhiên thứ tự."""
    rows = list(csv.DictReader(open(manifest_csv, encoding="utf-8")))
    rng = random.Random(seed)
    items = []
    for label in ("healthy", "rotten"):
        pool = [r for r in rows if r["fruit"] == fruit and r["split"] == split and r["label"] == label]
        rng.shuffle(pool)
        items += [dict(path=os.path.join(data_dir, r["path"]), fruit=fruit, label=label) for r in pool[:n_per_label]]
    rng.shuffle(items)
    return items


def make_conveyor_video(items, out_path, gt_path=None, cfg=None, seed=0):
    """
    Input : items = [dict(path, fruit, label), ...] theo thứ tự đi vào từ trái
            out_path = file .mp4 ; gt_path mặc định <out>_gt.csv
    Output: ghi video + ground-truth CSV (cột: idx,path,fruit,label,x0,y,speed,size)
            Vị trí cạnh trái quả i tại frame t:  x = x0_i + speed * t
    """
    cfg = cfg or config.VIDEO
    W, H, size = cfg["width"], cfg["height"], cfg["fruit_size"]
    speed, gap, fps = cfg["speed_px"], cfg["gap_px"], cfg["fps"]
    gt_path = gt_path or os.path.splitext(out_path)[0] + "_gt.csv"
    rng = random.Random(seed)

    imgs, centers = [], []
    for it in items:
        im = letterbox(imread(it["path"]), size)
        if cfg.get("flatten_bg", True):
            im, c = flatten_background(im)
        else:
            im, c = im.astype(np.float32), np.median(im.reshape(-1, 3), axis=0)
        imgs.append(im); centers.append(c)
    belt_cfg = cfg.get("belt_color")
    belt = np.array(belt_cfg if belt_cfg is not None else np.median(np.array(centers), axis=0), np.float32)
    alpha = feathered_alpha(size, cfg["feather"])[..., None]
    step = size + gap
    x0 = [-size - i * step for i in range(len(items))]
    ys = [(H - size) // 2 + rng.randint(-cfg["y_jitter"], cfg["y_jitter"]) for _ in items]
    n_frames = int((W - x0[-1]) / speed) + 2

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise IOError(f"Không tạo được video: {out_path}")
    for t in range(n_frames):
        frame = np.tile(belt, (H, W, 1))
        for i, img in enumerate(imgs):
            x = x0[i] + speed * t
            if x + size <= 0 or x >= W:
                continue
            xs, xe = max(x, 0), min(x + size, W)
            sub = img[:, xs - x:xe - x]
            a = alpha[:, xs - x:xe - x]
            roi = frame[ys[i]:ys[i] + size, xs:xe]
            frame[ys[i]:ys[i] + size, xs:xe] = roi * (1 - a) + sub * a
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()

    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "path", "fruit", "label", "x0", "y", "speed", "size"])
        for i, it in enumerate(items):
            w.writerow([i, it["path"], it["fruit"], it["label"], x0[i], ys[i], speed, size])
    print(f"Video: {out_path} ({n_frames} frame, {n_frames / fps:.0f}s, {len(items)} quả, màu băng chuyền BGR={tuple(int(x) for x in belt)})\nGround truth: {gt_path}")
    return out_path, gt_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", default="apple")
    ap.add_argument("--split", default="test", choices=["test", "calib"])
    ap.add_argument("--n", type=int, default=20, help="số quả mỗi nhãn (healthy/rotten)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out or os.path.join(config.DATA_DIR, "videos", f"{a.fruit}_{a.split}.mp4")
    its = select_from_manifest(os.path.join(config.DATA_DIR, "manifest.csv"), config.DATA_DIR,
                               a.fruit, a.split, a.n, a.seed)
    make_conveyor_video(its, out, seed=a.seed)
