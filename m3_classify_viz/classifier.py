"""
MODULE 3 - PHÂN LOẠI CHẤT LƯỢNG (Chương 0, 8): Rule-based Decision Matrix
    LOAI 3 (Khong dat): defect > g2_max_defect  HOAC  circularity < g3_max_circ  HOAC  ripe_ratio < g3_max_ripe
    LOAI 1 (Dat)      : defect < g1_max_defect  VA    circularity >= g1_min_circ  VA    ripe_ratio >= g1_min_ripe
    LOAI 2 (Dat)      : các trường hợp còn lại
Ngưỡng mặc định ở config.THRESHOLDS; nếu có outputs/thresholds_<fruit>.json (do calibrate.py tạo) thì ghi đè.
"""
import json
import os
from collections import Counter

import config
from common.contracts import GRADE_1, GRADE_2, GRADE_3


def load_thresholds(fruit=None):
    th = dict(config.THRESHOLDS)
    if fruit:
        p = os.path.join(config.OUT_DIR, f"thresholds_{fruit}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                th.update({k: v for k, v in json.load(f).items() if k in th})
    return th


def classify_rule_based(features, thresholds=None):
    """
    Input : feature dict (cần defect_ratio, circularity, ripe_ratio)
    Output: (label, reason) ; label thuộc {GRADE_1, GRADE_2, GRADE_3}
    """
    t = thresholds or config.THRESHOLDS
    d, c, r = features["defect_ratio"], features["circularity"], features["ripe_ratio"]

    bad = []
    if d > t["g2_max_defect"]:
        bad.append(f"defect {d:.1f}%>{t['g2_max_defect']:g}%")
    if c < t["g3_max_circ"]:
        bad.append(f"circ {c:.2f}<{t['g3_max_circ']:g}")
    if r < t["g3_max_ripe"]:
        bad.append(f"ripe {r:.2f}<{t['g3_max_ripe']:g}")
    if bad:
        return GRADE_3, "; ".join(bad)

    weak = []
    if d >= t["g1_max_defect"]:
        weak.append(f"defect {d:.1f}%>={t['g1_max_defect']:g}%")
    if c < t["g1_min_circ"]:
        weak.append(f"circ {c:.2f}<{t['g1_min_circ']:g}")
    if r < t["g1_min_ripe"]:
        weak.append(f"ripe {r:.2f}<{t['g1_min_ripe']:g}")
    if weak:
        return GRADE_2, "; ".join(weak)
    return GRADE_1, "dat chuan loai 1"


def majority_grade(labels):
    """Bỏ phiếu đa số; nếu hoà thì chọn hạng XẤU hơn (an toàn cho kiểm định chất lượng)."""
    cnt = Counter(labels)
    top = max(cnt.values())
    order = [GRADE_3, GRADE_2, GRADE_1]
    return next(g for g in order if cnt.get(g, 0) == top)
