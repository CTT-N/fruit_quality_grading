"""
MODULE 3 - PIPELINE VIDEO ĐẦY ĐỦ:  main.run_pipeline
   đọc frame -> M1 (tiền xử lý, tách nền, tách quả, tracking) -> M2 (đặc trưng) -> luật phân loại -> vẽ -> ghi video

Mỗi quả chỉ được đo trong "vùng quyết định" ở giữa khung, khi quả hiện đủ hình (không bị cắt mép),
đo vote_frames lần rồi bỏ phiếu đa số -> nhãn cuối của quả đó.
"""
import csv
import os
import time
from collections import Counter

import numpy as np

import config
from common.contracts import GRADE_1, GRADE_2, GRADE_3, is_pass
from m1_segmentation.pipeline_m1 import detect_fruits
from m1_segmentation.objects import is_fully_visible
from m1_segmentation.tracker import CentroidTracker
from m2_features.extract_all import extract_all_features
from .classifier import classify_rule_based, majority_grade, load_thresholds
from .video_io import get_video_info, read_video_frames, create_writer
from .visualize import draw_overlay

SCALARS = ("defect_ratio", "n_spots", "circularity", "ripe_ratio", "d_e_px", "area_px")
CSV_FIELDS = ["track_id", "frame", "cx", "cy", "label", "pass", "defect_ratio", "n_spots",
              "circularity", "ripe_ratio", "d_e_px", "area_px", "reason"]


class _Track:
    def __init__(self):
        self.labels, self.feats, self.reasons = [], [], []
        self.first = None                 # (frame, cx, cy) lần đo đầu tiên
        self.done, self.label, self.defect_mask, self.frames_seen = False, None, None, 0

    def vote(self, frame_idx, o, feat, label, reason):
        if self.first is None:
            self.first = (frame_idx, o.centroid[0], o.centroid[1])
        self.labels.append(label); self.reasons.append(reason)
        self.feats.append({k: float(feat[k]) for k in SCALARS})
        self.defect_mask = feat["defect_mask"]
        self.label = majority_grade(self.labels)

    def summary(self):
        med = {k: float(np.median([f[k] for f in self.feats])) for k in SCALARS}
        reason = next((r for l, r in zip(self.labels, self.reasons) if l == self.label), "")
        return med, reason


def run_pipeline(video_in, video_out, fruit="apple", results_csv=None, show=False, max_frames=None, verbose=True):
    """
    Input : video_in (.mp4), video_out (.mp4), fruit 'apple'|'orange'
    Output: dict(results=list[dict], fps=float, n_frames=int, results_csv=str)
    """
    info = get_video_info(video_in)
    W, H = info["width"], info["height"]
    zone_px = (config.PIPELINE["zone"][0] * W, config.PIPELINE["zone"][1] * W)
    K, step = config.PIPELINE["vote_frames"], config.PIPELINE["vote_step"]
    th = load_thresholds(fruit)
    results_csv = results_csv or os.path.splitext(video_out)[0] + "_results.csv"

    tracker, tracks, counts, rows = CentroidTracker(), {}, Counter({GRADE_1: 0, GRADE_2: 0, GRADE_3: 0}), []
    writer = create_writer(video_out, info["fps"], (W, H))
    t_start, fps_run, n = time.time(), 0.0, 0

    def finalize(tid, st):
        med, reason = st.summary()
        counts[st.label] += 1
        st.done = True
        rows.append(dict(track_id=tid, frame=st.first[0], cx=round(st.first[1], 1), cy=round(st.first[2], 1),
                         label=st.label, **{"pass": int(is_pass(st.label))},
                         **{k: round(v, 3) for k, v in med.items()}, reason=reason))

    for idx, frame in read_video_frames(video_in):
        if max_frames and idx >= max_frames:
            break
        t0 = time.time()
        objs, _ = detect_fruits(frame, idx)
        tracker.update(objs)
        infos = {}
        for o in objs:
            st = tracks.setdefault(o.track_id, _Track())
            st.frames_seen += 1
            cx = o.centroid[0]
            if (not st.done and len(st.labels) < K and zone_px[0] <= cx <= zone_px[1]
                    and is_fully_visible(o, frame.shape) and idx % step == 0):
                feat = extract_all_features(o, fruit)
                label, reason = classify_rule_based(feat, th)
                st.vote(idx, o, feat, label, reason)
            if not st.done and st.labels and (len(st.labels) >= K or cx > zone_px[1]):
                finalize(o.track_id, st)
            if st.labels:
                med, _ = st.summary()
                infos[o.track_id] = dict(label=st.label, defect_ratio=med["defect_ratio"],
                                         circularity=med["circularity"], defect_mask=st.defect_mask)
        fps_run = 0.9 * fps_run + 0.1 / max(time.time() - t0, 1e-6) if n else 1.0 / max(time.time() - t0, 1e-6)
        out = draw_overlay(frame, objs, infos, counts, fps_run, idx, zone_px)
        writer.write(out)
        n += 1
        if show:
            import cv2
            cv2.imshow("Fruit grading", out)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if verbose and n % 200 == 0:
            print(f"  ... frame {n}/{info['n_frames']}")
    writer.release()
    if show:
        import cv2; cv2.destroyAllWindows()

    for tid, st in tracks.items():                        # quả chưa chốt khi hết video
        if not st.done and st.labels:
            finalize(tid, st)
    no_vote = [tid for tid, st in tracks.items() if not st.labels and st.frames_seen >= 5]

    os.makedirs(os.path.dirname(os.path.abspath(results_csv)), exist_ok=True)
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: r["frame"]))
    elapsed = time.time() - t_start
    avg_fps = n / elapsed if elapsed > 0 else 0.0
    if verbose:
        print(f"\nXử lý {n} frame trong {elapsed:.1f}s  ->  {avg_fps:.1f} FPS trung bình")
        print(f"Số quả đã phân loại: {len(rows)}  (L1={counts[GRADE_1]}, L2={counts[GRADE_2]}, L3={counts[GRADE_3]})")
        if no_vote:
            print(f"[CẢNH BÁO] {len(no_vote)} track không được đo lần nào (id: {no_vote})")
        print(f"Video: {video_out}\nKết quả: {results_csv}")
    return dict(results=rows, fps=avg_fps, n_frames=n, results_csv=results_csv, no_vote=no_vote)
