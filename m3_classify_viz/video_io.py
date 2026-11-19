"""MODULE 3 - ĐỌC / GHI VIDEO (Chương 1, 4)"""
import cv2


def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {path}")
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def read_video_frames(path):
    """Generator: yield (frame_idx, frame_bgr). Mỗi frame là một ma trận f(x,y) (Ch.1)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def create_writer(path, fps, size_wh):
    """VideoWriter mp4v. size_wh = (width, height)."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size_wh)
    if not w.isOpened():
        raise IOError(f"Không tạo được file video: {path}")
    return w
