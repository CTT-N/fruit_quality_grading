"""
MODULE 3 - HIỆU CHỈNH NGƯỠNG & ĐÁNH GIÁ TRÊN ẢNH TĨNH (Chương 0, 8)

    python -m m3_classify_viz.calibrate --fruit apple --source video  # KHUYÊN DÙNG: hiệu chỉnh trên video calib (cùng miền dữ liệu với lúc chạy thật)
    python -m m3_classify_viz.calibrate --fruit apple                 # hiệu chỉnh trên ảnh tĩnh (nhanh, nhưng dễ lệch miền so với video)
    python -m m3_classify_viz.calibrate --fruit apple --split test    # tập TEST: đánh giá với ngưỡng đã lưu (KHÔNG chỉnh gì)
Nguyên tắc: ngưỡng chỉ được chọn trên tập calib; tập test chỉ dùng để báo cáo kết quả (tránh Overfitting, Ch.8).
Kết quả: outputs/features_<fruit>_<split>.csv, outputs/thresholds_<fruit>.json, outputs/figures/*.png
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from common.utils import imread
from m1_segmentation.pipeline_m1 import detect_fruits
from m2_features.extract_all import extract_all_features
from .classifier import classify_rule_based, load_thresholds
from .evaluate import binary_metrics, print_metrics, plot_confusion

COLS = ["defect_ratio", "n_spots", "circularity", "ripe_ratio", "d_e_px", "area_px",
        "aspect_ratio", "mean_h", "mean_s", "mean_v"]


def image_features(path, fruit):
    """Đặc trưng của vật LỚN NHẤT trong ảnh tĩnh. Trả về (feature dict | None, số vật)."""
    objs, _ = detect_fruits(imread(path))
    if not objs:
        return None, 0
    o = max(objs, key=lambda x: int((x.mask > 0).sum()))
    return extract_all_features(o, fruit), len(objs)


def build_table(fruit, split):
    manifest = list(csv.DictReader(open(os.path.join(config.DATA_DIR, "manifest.csv"), encoding="utf-8")))
    rows = []
    for r in [r for r in manifest if r["fruit"] == fruit and r["split"] == split]:
        feat, n_obj = image_features(os.path.join(config.DATA_DIR, r["path"]), fruit)
        if feat is None:
            print(f"[BỎ QUA] không tách được quả: {r['path']}")
            continue
        rows.append(dict(path=r["path"], label=r["label"], n_objects=n_obj, **{c: float(feat[c]) for c in COLS}))
    out = os.path.join(config.OUT_DIR, f"features_{fruit}_{split}.csv")
    os.makedirs(config.OUT_DIR, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Đã lưu đặc trưng {len(rows)} ảnh: {out}")
    return rows


def build_table_from_video(fruit):
    """
    Tạo video từ tập CALIB, chạy pipeline thật (M1->M2), ghép kết quả với ground truth theo vị trí.
    Lý do: nén video + trộn nền làm phát sinh nhiễu mà ảnh tĩnh không có -> ngưỡng phải học trên cùng miền dữ liệu.
    """
    from .make_video import select_from_manifest, make_conveyor_video
    from .pipeline import run_pipeline
    from .evaluate import match_results_to_gt
    vid = os.path.join(config.DATA_DIR, "videos", f"{fruit}_calib.mp4")
    items = select_from_manifest(os.path.join(config.DATA_DIR, "manifest.csv"), config.DATA_DIR, fruit, "calib", 10**6)
    _, gt_path = make_conveyor_video(items, vid)
    res = run_pipeline(vid, os.path.join(config.OUT_DIR, f"{fruit}_calib_result.mp4"), fruit, verbose=False)
    gt = list(csv.DictReader(open(gt_path, encoding="utf-8")))
    pairs, dup, missed = match_results_to_gt(res["results"], gt)
    print(f"Video calib: ghép {len(pairs)}/{len(gt)} quả (trùng={dup}, bỏ sót={missed})")
    rows = [dict(path=g["path"], label=g["label"], n_objects=1,
                 **{k: float(r[k]) for k in ("defect_ratio", "n_spots", "circularity", "ripe_ratio", "d_e_px", "area_px")})
            for r, g in pairs]
    out = os.path.join(config.OUT_DIR, f"features_{fruit}_calib_video.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("Đã lưu đặc trưng (video):", out)
    return rows


def _predict_fail(rows, th):
    return np.array([classify_rule_based(r, th)[0] == "LOAI 3" for r in rows])


def search_thresholds(rows, base):
    """Tìm (g2_max_defect, g3_max_ripe) cho balanced accuracy cao nhất trên tập calib."""
    y = np.array([r["label"] == "rotten" for r in rows])
    d = np.array([r["defect_ratio"] for r in rows]); c = np.array([r["circularity"] for r in rows])
    rp = np.array([r["ripe_ratio"] for r in rows])
    best = (-1, None, None)
    for X in np.concatenate([np.arange(0.25, 2.0, 0.25), np.arange(2.0, 40.01, 0.5)]):
        for rr in np.arange(0.0, 0.61, 0.05):
            pf = (d > X) | (c < base["g3_max_circ"]) | (rp < rr)
            tpr = (pf & y).sum() / max(y.sum(), 1)
            tnr = (~pf & ~y).sum() / max((~y).sum(), 1)
            if (tpr + tnr) / 2 > best[0] + 1e-9:
                best = ((tpr + tnr) / 2, float(X), float(rr))
    return best


def plot_distributions(rows, fruit, path):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, key, ttl in zip(ax, ["defect_ratio", "circularity", "ripe_ratio"],
                           ["Ti le khuyet tat (%)", "Circularity", "Ti le mau chin (ripe_ratio)"]):
        for lab, col in (("healthy", "tab:green"), ("rotten", "tab:red")):
            v = [r[key] for r in rows if r["label"] == lab]
            a.hist(v, bins=15, alpha=0.6, color=col, label=lab)
        a.set_title(ttl, fontsize=10); a.legend(fontsize=8)
    fig.suptitle(f"Phan bo dac trung - {fruit}"); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", default="apple", choices=list(config.FRUIT_CFG))
    ap.add_argument("--split", default="calib", choices=["calib", "test"])
    ap.add_argument("--source", default="image", choices=["image", "video"],
                    help="video: hiệu chỉnh trên video calib (khuyên dùng); chỉ áp dụng cho split calib")
    a = ap.parse_args()

    rows = build_table_from_video(a.fruit) if (a.source == "video" and a.split == "calib") \
        else build_table(a.fruit, a.split)
    figdir = os.path.join(config.OUT_DIR, "figures")
    plot_distributions(rows, a.fruit, os.path.join(figdir, f"dist_{a.fruit}_{a.split}_{a.source}.png"))
    y_true = np.array([r["label"] == "rotten" for r in rows])

    if a.split == "calib":
        base = dict(config.THRESHOLDS)
        m0 = binary_metrics(y_true, _predict_fail(rows, base))
        print_metrics(m0, f"CALIB - ngưỡng mặc định (defect>{base['g2_max_defect']}%)")
        bal, X, rr = search_thresholds(rows, base)
        new = dict(base, g2_max_defect=X, g3_max_ripe=rr, g1_max_defect=min(base["g1_max_defect"], X / 2))   # Loại 1 luôn chặt hơn ngưỡng loại bỏ
        m1 = binary_metrics(y_true, _predict_fail(rows, new))
        print_metrics(m1, f"CALIB - ngưỡng tối ưu (defect>{X}%, ripe<{rr:.2f}), balanced acc={bal:.1%}")
        with open(os.path.join(config.OUT_DIR, f"thresholds_{a.fruit}.json"), "w", encoding="utf-8") as f:
            json.dump(new, f, indent=2)
        print(f"Đã lưu ngưỡng: outputs/thresholds_{a.fruit}.json  (classifier tự động dùng file này)")
        n_multi = sum(1 for r in rows if r["n_objects"] != 1)
        if n_multi:
            print(f"[LƯU Ý] {n_multi} ảnh tách ra != 1 vật (đã lấy vật lớn nhất) -> nên xem lại.")
    else:
        th = load_thresholds(a.fruit)
        m = binary_metrics(y_true, _predict_fail(rows, th))
        print_metrics(m, f"TEST - ngưỡng đã lưu {{defect>{th['g2_max_defect']}%, ripe<{th['g3_max_ripe']:.2f}}}")
        plot_confusion(m, os.path.join(figdir, f"confusion_{a.fruit}_test.png"), f"{a.fruit} - tap test (anh tinh)")
        wrong = [r for r, yt, yp in zip(rows, y_true, _predict_fail(rows, th)) if yt != yp]
        print(f"\n{len(wrong)} ảnh phân loại sai (dùng cho phần Failure Cases của báo cáo):")
        for r in wrong[:15]:
            print(f"  {r['path']:<55} defect={r['defect_ratio']:5.1f}% circ={r['circularity']:.2f} ripe={r['ripe_ratio']:.2f}")


if __name__ == "__main__":
    main()
