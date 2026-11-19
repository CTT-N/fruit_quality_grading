"""
CHẠY HỆ THỐNG TRÊN VIDEO:
    python main.py --video data/videos/apple_test.mp4 --fruit apple
    python main.py --video data/videos/orange_test.mp4 --fruit orange --show
Kết quả: outputs/<ten_video>_result.mp4, ..._results.csv, confusion matrix (nếu có file ground-truth cạnh video).
"""
import argparse
import os

import config
from m3_classify_viz.pipeline import run_pipeline
from m3_classify_viz.evaluate import evaluate_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--fruit", default="apple", choices=list(config.FRUIT_CFG))
    ap.add_argument("--out", default=None)
    ap.add_argument("--gt", default=None, help="ground_truth.csv (mặc định: <video>_gt.csv)")
    ap.add_argument("--show", action="store_true", help="hiển thị cửa sổ khi xử lý (ESC để thoát)")
    a = ap.parse_args()

    name = os.path.splitext(os.path.basename(a.video))[0]
    out = a.out or os.path.join(config.OUT_DIR, f"{name}_result.mp4")
    res = run_pipeline(a.video, out, a.fruit, show=a.show)
    gt = a.gt or os.path.splitext(a.video)[0] + "_gt.csv"
    if os.path.exists(gt):
        evaluate_video(res["results_csv"], gt,
                       os.path.join(config.OUT_DIR, "figures", f"confusion_{name}.png"), title=name)
    else:
        print(f"(Không thấy {gt} nên bỏ qua bước đánh giá)")


if __name__ == "__main__":
    main()
