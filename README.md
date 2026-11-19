# Hệ thống phân loại chất lượng trái cây (Xử lý ảnh cổ điển, không Deep Learning)

Mọi lệnh chạy **ở thư mục gốc dự án** (nơi có `config.py` và `main.py`).

## Cài đặt
```
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Trạng thái
| Phần | Người | Trạng thái |
|---|---|---|
| `common/`, `config.py` | cả nhóm | ✅ |
| `m1_segmentation/` (chuẩn hoá dataset, tiền xử lý, tách nền, tracking) | TV1 | ✅ |
| `m2_features/` (màu sắc, khuyết tật, hình dạng) | TV2 | ✅ |
| `m3_classify_viz/` + `main.py` (video, phân loại, hiển thị, đánh giá) | TV3 | ✅ |

## Test tự động (không cần dataset thật)
```
python -m m1_segmentation.test_m1
python -m m2_features.test_m2
python -m m3_classify_viz.test_m3      # test tích hợp cả hệ thống trên video giả lập
```

## Quy trình chạy trên dữ liệu thật (theo thứ tự)
Dữ liệu đặt ở `data/raw/Apple/{Fresh,Rotten}` và `data/raw/Orange/{Fresh,Rotten}`.

| Bước | Lệnh | Ghi chú |
|---|---|---|
| 1 | `python -m m1_segmentation.curate_dataset` | Chọn 20 ảnh calib + 30 ảnh test mỗi (quả, nhãn), letterbox 320x320 -> `data/calib`, `data/test`, `data/manifest.csv` |
| 2 | `python -m m1_segmentation.test_m1 --folder data/calib/apple/rotten --n 12` | Kiểm tra mắt thường mặt nạ tách nền (TV1) |
| 3 | `python -m m2_features.test_m2 --image data/calib/apple/rotten/apple_rotten_calib_002.png --fruit apple` (tên file đúng dạng `<quả>_<nhãn>_calib_<số>.png`, xem trong `data/calib/`) | Kiểm tra phát hiện khuyết tật trên ảnh thật (TV2) |
| 4 | `python -m m3_classify_viz.calibrate --fruit apple --source video` | **Hiệu chỉnh ngưỡng trên video calib** -> `outputs/thresholds_apple.json` |
| 5 | `python -m m3_classify_viz.make_video --fruit apple --split test --n 15` | Tạo `data/videos/apple_test.mp4` + `apple_test_gt.csv` (30 quả) |
| 6 | `python main.py --video data/videos/apple_test.mp4 --fruit apple` | Chạy hệ thống: video kết quả + CSV + confusion matrix + FPS |
| 7 | Lặp bước 4-6 với `--fruit orange` | |
| 8 | `python -m m3_classify_viz.calibrate --fruit apple --split test` | (tuỳ chọn) đánh giá trên ảnh tĩnh, liệt kê ảnh sai cho phần Failure Cases |

Nguyên tắc: **chỉ chỉnh ngưỡng bằng tập calib**; tập test chỉ để báo cáo (tránh Overfitting, Ch.8).
Nên dùng `--source video` ở bước 4: nén video và trộn nền tạo nhiễu mà ảnh tĩnh không có, nên ngưỡng học trên ảnh tĩnh có thể quá nhạy khi chạy video.

## Quy ước chung
- Chỉ sửa trong thư mục module của mình; `common/` chỉ sửa khi cả nhóm đồng ý.
- Mọi ngưỡng/tham số nằm ở `config.py`; ngưỡng phân loại được `calibrate.py` ghi đè qua `outputs/thresholds_<fruit>.json`.
- Nhãn ASCII (`LOAI 1/2/3`) vì `cv2.putText` không vẽ được chữ có dấu.
- Đánh giá: Loại 1, 2 = ĐẠT; Loại 3 = KHÔNG ĐẠT; so với nhãn dataset (Fresh -> đạt, Rotten -> không đạt). Bỏ `Formalin-mixed`.

## Chỉnh khi gặp lỗi (config.py)
| Hiện tượng | Chỉnh |
|---|---|
| Tách nền sai/thiếu quả | `SEGMENT.min_dist/max_dist`, thử `SEGMENT.method = "otsu_s"` hoặc `"kmeans"`; nền bị tối dần ra góc (vignette) đã được xử lý bởi `bg_model="surface"` (mặc định) |
| Quả lành báo có vết (rìa/bóng đổ) | `DEFECT.d0_min`, `DEFECT.erode_ratio` ↑ |
| Bỏ sót vết thâm mờ | `DEFECT.d0_min` ↓ |
| Đếm quá nhiều vết nhỏ | `DEFECT.min_spot_area_ratio`, `DEFECT.open_k` ↑ |
| Điểm chói bị tính là lỗi | `DEFECT.highlight_s_max/v_min` |
| Dải màu "đạt" sai | `FRUIT_CFG[...]['ripe_hue_ranges']` |
| Video chạy chậm | `PIPELINE.vote_step` ↑, giảm `VIDEO.width/height` |

## Hạn chế cần ghi vào báo cáo (Failure Cases)
- Vết hỏng cùng màu vỏ hoặc rất mờ sẽ bị bỏ sót; vết rất nhỏ khó phân biệt với nhiễu nén video.
- Vùng hỏng > ~50% quả và cùng dải màu "đạt" có thể làm chọn nhầm màu lành.
- Nấm mốc trắng nhạt có thể bị loại cùng điểm chói phản quang.
- Circularity chỉ nhạy khi quả méo mạnh (tỉ lệ trục ~2:1 mới xuống ~0,75).
- Video là mô phỏng (băng chuyền nhẵn, tốc độ đều, quả cách đều); kích thước S/M/L chỉ mang tính tương đối.
- Ngưỡng 1%/10% theo giáo trình không phù hợp mọi bộ dữ liệu: cần hiệu chỉnh (calibrate) và nêu rõ trong báo cáo.
