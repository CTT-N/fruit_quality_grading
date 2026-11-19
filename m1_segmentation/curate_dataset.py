"""
MODULE 1 - CHỌN VÀ CHUẨN HOÁ DATASET (Chương 1, 4)

Đọc data/raw/<Fruit>/<Fresh|Rotten>/*.jpg  ->  ghi ra
    data/calib/<fruit>/<healthy|rotten>/*.png   (tập hiệu chỉnh ngưỡng)
    data/test/<fruit>/<healthy|rotten>/*.png    (tập kiểm thử, KHÔNG dùng để chỉnh ngưỡng)
    data/manifest.csv                            (đường dẫn, loại quả, nhãn thật, split)

Chuẩn hoá: letterbox giữ tỉ lệ về canvas vuông (không méo -> Circularity không bị sai),
phần đệm lấy từ pixel biên ảnh (BORDER_REPLICATE) để không tạo vùng màu giả; lưu PNG lossless (Ch.4).
Bỏ qua thư mục Formalin-mixed và các file không phải ảnh (vd desktop.ini).

Chạy ở thư mục gốc dự án:
    python -m m1_segmentation.curate_dataset
    python -m m1_segmentation.curate_dataset --n_calib 20 --n_test 30 --size 320
"""
import argparse
import csv
import os
import random

import cv2

import config
from common.utils import imread, imwrite

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
LABEL_MAP = {"fresh": "healthy", "rotten": "rotten"}   # tên thư mục raw -> nhãn thật


def letterbox(img, size=320):
    """
    Resize giữ tỉ lệ sao cho cạnh dài = size, đệm phần còn lại bằng pixel biên (BORDER_REPLICATE).
    Input : ảnh BGR (h,w,3) uint8 ; Output: ảnh BGR (size,size,3) uint8
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))


def curate_dataset(raw_dir, out_dir, fruits, n_calib, n_test, size, seed=0):
    """
    Với mỗi (loại quả, nhãn): lấy ngẫu nhiên n_calib + n_test ảnh, chia 2 tập KHÔNG giao nhau.
    Trả về danh sách dòng manifest.
    """
    rng = random.Random(seed)
    rows = []
    for fruit in fruits:
        for raw_cls, label in LABEL_MAP.items():
            # tìm thư mục không phân biệt hoa/thường (Fresh / fresh)
            fruit_dir = _find_dir(raw_dir, fruit)
            cls_dir = _find_dir(fruit_dir, raw_cls) if fruit_dir else None
            if cls_dir is None:
                print(f"[CẢNH BÁO] Không thấy thư mục {fruit}/{raw_cls} trong {raw_dir}")
                continue
            files = list_images(cls_dir)
            rng.shuffle(files)
            need = n_calib + n_test
            if len(files) < need:
                print(f"[CẢNH BÁO] {fruit}/{raw_cls} chỉ có {len(files)} ảnh (cần {need}); "
                      f"chia theo tỉ lệ.")
            n_c = round(len(files) * n_calib / need) if len(files) < need else n_calib
            splits = {"calib": files[:n_c], "test": files[n_c:n_c + n_test]}
            for split, names in splits.items():
                for i, name in enumerate(names):
                    try:
                        img = letterbox(imread(os.path.join(cls_dir, name)), size)
                    except Exception as e:        # ảnh hỏng -> bỏ qua
                        print(f"[BỎ QUA] {name}: {e}")
                        continue
                    out_name = f"{fruit.lower()}_{label}_{split}_{i:03d}.png"
                    rel = os.path.join(split, fruit.lower(), label, out_name)
                    imwrite(os.path.join(out_dir, rel), img)
                    rows.append(dict(path=rel.replace("\\", "/"), fruit=fruit.lower(),
                                     label=label, split=split, original=name))
            print(f"{fruit:<7} {raw_cls:<7} -> calib={len(splits['calib'])}, test={len(splits['test'])}")
    return rows


def _find_dir(parent, name):
    if parent is None or not os.path.isdir(parent):
        return None
    for d in os.listdir(parent):
        if d.lower() == name.lower() and os.path.isdir(os.path.join(parent, d)):
            return os.path.join(parent, d)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(config.DATA_DIR, "raw"))
    ap.add_argument("--out", default=config.DATA_DIR)
    ap.add_argument("--fruits", nargs="+", default=["Apple", "Orange"])
    ap.add_argument("--n_calib", type=int, default=20, help="số ảnh calib mỗi (quả, nhãn)")
    ap.add_argument("--n_test", type=int, default=30, help="số ảnh test mỗi (quả, nhãn)")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = curate_dataset(a.raw, a.out, a.fruits, a.n_calib, a.n_test, a.size, a.seed)
    manifest = os.path.join(a.out, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "fruit", "label", "split", "original"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nTổng {len(rows)} ảnh. Manifest: {manifest}")


if __name__ == "__main__":
    main()
