"""Hàm tiện ích dùng chung (đọc/ghi ảnh an toàn đường dẫn Unicode, ghép ảnh để minh hoạ)."""
import os
import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread nhưng đọc được đường dẫn có dấu tiếng Việt / khoảng trắng trên Windows."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return img


def imwrite(path, img):
    """cv2.imwrite an toàn Unicode. Dùng PNG cho mặt nạ (lossless, Ch.4)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Không ghi được ảnh: {path}")
    buf.tofile(path)


def to_bgr(img):
    """Gray/mask -> BGR để ghép montage."""
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def montage(images, titles=None, height=300):
    """Ghép nhiều ảnh thành 1 dải ngang (dùng để chụp ảnh minh hoạ cho báo cáo)."""
    tiles = []
    for i, im in enumerate(images):
        im = to_bgr(im)
        h, w = im.shape[:2]
        im = cv2.resize(im, (int(w * height / h), height))
        if titles is not None:
            cv2.putText(im, titles[i], (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(im)
    return np.hstack(tiles)
