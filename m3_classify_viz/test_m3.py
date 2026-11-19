"""
TEST MODULE 3 + TÍCH HỢP TOÀN HỆ THỐNG (chạy ở thư mục gốc):
    python -m m3_classify_viz.test_m3
Bao gồm: kiểm tra luật phân loại, tạo video giả lập từ quả mock, chạy full pipeline M1->M2->M3 và đánh giá.
Không cần dataset thật.
"""
import os
import tempfile

import cv2

import config
from common.contracts import GRADE_1, GRADE_2, GRADE_3
from m1_segmentation.test_m1 import build_still
from common.utils import imwrite
from .classifier import classify_rule_based, majority_grade
from .make_video import make_conveyor_video
from .pipeline import run_pipeline
from .evaluate import evaluate_video

BGS = [(232, 230, 228), (244, 232, 218)]          # xám-trắng, lam nhạt-trắng


def test_classifier():
    ok = dict(circularity=0.95, ripe_ratio=0.95)
    assert classify_rule_based(dict(defect_ratio=0.2, **ok))[0] == GRADE_1
    assert classify_rule_based(dict(defect_ratio=4.0, **ok))[0] == GRADE_2
    assert classify_rule_based(dict(defect_ratio=25.0, **ok))[0] == GRADE_3
    assert classify_rule_based(dict(defect_ratio=0.2, circularity=0.6, ripe_ratio=0.9))[0] == GRADE_3
    assert classify_rule_based(dict(defect_ratio=0.2, circularity=0.8, ripe_ratio=0.9))[0] == GRADE_2
    assert majority_grade([GRADE_1, GRADE_3, GRADE_3]) == GRADE_3
    assert majority_grade([GRADE_1, GRADE_3]) == GRADE_3            # hoà -> chọn hạng xấu hơn
    print("[OK] luật phân loại + bỏ phiếu")


def make_synthetic_items(tmp, n_each=5):
    """Tạo ảnh tĩnh 320x320 giống dataset thật (quả có lề, nền xám/lam nhạt tối dần ra góc); trả items cho make_conveyor_video."""
    items = []
    for i in range(n_each * 2):
        healthy = i % 2 == 0
        kw = dict(kind=["apple_red", "apple_green", "apple_red"][i % 3])
        if not healthy:
            kw.update(n_spots=2 + i % 2, spot_radius=22) if i % 4 == 1 else kw.update(rot_fraction=0.35, n_spots=1)
        img, _ = build_still(140 + 15 * (i % 4), vg=0.05 * (i % 5), bg=BGS[i % 2], seed=i, **kw)
        p = os.path.join(tmp, f"img_{i}.png")
        imwrite(p, img)
        items.append(dict(path=p, fruit="apple", label="healthy" if healthy else "rotten"))
    return items


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        items = make_synthetic_items(tmp, n_each=5)
        vid = os.path.join(tmp, "synthetic.mp4")
        make_conveyor_video(items, vid)
        out = os.path.join(tmp, "result.mp4")
        res = run_pipeline(vid, out, fruit="apple", verbose=True)
        assert len(res["results"]) == len(items), f"phân loại {len(res['results'])}/{len(items)} quả"
        assert not res["no_vote"], "có quả không được đo lần nào"
        assert os.path.getsize(out) > 10_000, "video output rỗng"
        m = evaluate_video(res["results_csv"], os.path.splitext(vid)[0] + "_gt.csv",
                           os.path.join(config.OUT_DIR, "figures", "confusion_synthetic.png"), "synthetic")
        assert m["n_matched"] == len(items), "ghép kết quả - ground truth thiếu"
        assert m["accuracy"] >= 0.9, f"accuracy {m['accuracy']:.1%} < 90%"
        # lưu 1 frame minh hoạ giữa video
        cap = cv2.VideoCapture(out); cap.set(cv2.CAP_PROP_POS_FRAMES, 120); ok, fr = cap.read(); cap.release()
        if ok:
            imwrite(os.path.join(config.OUT_DIR, "figures", "m3_demo_frame.png"), fr)
    print("[OK] tích hợp M1 -> M2 -> M3 trên video")


if __name__ == "__main__":
    test_classifier()
    test_end_to_end()
    print("\n>>> TẤT CẢ TEST MODULE 3 ĐẠT")
