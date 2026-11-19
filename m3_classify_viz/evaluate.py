"""
MODULE 3 - ĐÁNH GIÁ (Chương 0: đánh giá định lượng, confusion matrix, FPS)
Quy ước: ground truth 'healthy' kỳ vọng ĐẠT (Loại 1/2), 'rotten' kỳ vọng KHÔNG ĐẠT (Loại 3).
Dương tính (positive) = KHÔNG ĐẠT (phát hiện quả hỏng).
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config


def binary_metrics(y_true_fail, y_pred_fail):
    """Input: 2 mảng bool (True = không đạt). Output: dict tp,fp,tn,fn,accuracy,precision,recall,f1."""
    yt, yp = np.asarray(y_true_fail, bool), np.asarray(y_pred_fail, bool)
    tp, fp = int((yt & yp).sum()), int((~yt & yp).sum())
    tn, fn = int((~yt & ~yp).sum()), int((yt & ~yp).sum())
    n = max(len(yt), 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, accuracy=(tp + tn) / n, precision=prec, recall=rec, f1=f1)


def plot_confusion(m, path, title="Confusion matrix"):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    names = ["DAT", "KHONG DAT"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(names); ax.set_yticks([0, 1]); ax.set_yticklabels(names)
    ax.set_xlabel("Du doan"); ax.set_ylabel("Thuc te (nhan dataset)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"{title}\nAcc={m['accuracy']:.1%}  Recall(hong)={m['recall']:.1%}", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


def print_metrics(m, title=""):
    print(f"\n=== {title} ===")
    print(f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    print(f"Accuracy={m['accuracy']:.1%}  Precision={m['precision']:.1%}  Recall={m['recall']:.1%}  F1={m['f1']:.3f}")


def match_results_to_gt(results, gt_rows):
    """
    Ghép mỗi quả phân loại được với quả thật gần nhất theo VỊ TRÍ tại frame quyết định:
        cx_gt(i, f) = x0_i + speed * f + size/2
    Trả về (pairs [(result, gt)], n_duplicate, n_missed)
    """
    pairs, used, dup = [], set(), 0
    for r in results:
        f, cx = int(r["frame"]), float(r["cx"])
        best, bd = None, 1e18
        for g in gt_rows:
            gcx = float(g["x0"]) + float(g["speed"]) * f + float(g["size"]) / 2
            if abs(cx - gcx) < bd:
                best, bd = g, abs(cx - gcx)
        if best is None or bd > float(best["size"]) / 2:
            continue
        if int(best["idx"]) in used:
            dup += 1
            continue
        used.add(int(best["idx"]))
        pairs.append((r, best))
    return pairs, dup, len(gt_rows) - len(used)


def evaluate_video(results_csv, gt_csv, fig_path=None, title="Video test"):
    results = list(csv.DictReader(open(results_csv, encoding="utf-8")))
    gt = list(csv.DictReader(open(gt_csv, encoding="utf-8")))
    pairs, dup, missed = match_results_to_gt(results, gt)
    y_true = [g["label"] == "rotten" for _, g in pairs]
    y_pred = [r["label"] == "LOAI 3" for r, _ in pairs]
    m = binary_metrics(y_true, y_pred)
    print_metrics(m, f"{title}  ({len(pairs)}/{len(gt)} quả ghép được, trùng={dup}, bỏ sót={missed})")
    # phân bố hạng theo nhãn thật
    for lab in ("healthy", "rotten"):
        cnt = {g: sum(1 for r, gg in pairs if gg["label"] == lab and r["label"] == g)
               for g in ("LOAI 1", "LOAI 2", "LOAI 3")}
        print(f"  {lab:<8}: {cnt}")
    if fig_path:
        plot_confusion(m, fig_path, title)
        print("Đã lưu confusion matrix:", fig_path)
    m.update(n_matched=len(pairs), n_gt=len(gt), n_dup=dup, n_missed=missed)
    return m
