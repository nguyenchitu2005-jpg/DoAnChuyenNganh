#!/usr/bin/env bash
# Phương án A (kế hoạch T1): sau khi `python src/preprocess.py --out data/clips --clip-len 8` xong,
# chạy nốt mọi bước trên máy (không dùng Kaggle). Chạy: bash run_temporal.sh  (log: runs/temporal/pipeline.log)
# Bước nào đã có kết quả thì tự bỏ qua, nên chạy lại an toàn.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
mkdir -p runs/temporal
exec > >(tee -a runs/temporal/pipeline.log) 2>&1
echo "=== $(date '+%F %T') bắt đầu"

# 0) bù video c23 còn sót (video đã có trong log được bỏ qua)
$PY src/preprocess.py --out data/clips --clip-len 8
# 1) đoạn khung c40: cùng khung và cùng khung cắt với c23
$PY src/preprocess.py --compression c40 --out data/clips_c40 --boxes-from data/clips --workers 4
# 2) index các đoạn
[ -f data/index_clips/train.csv ] || $PY src/build_index.py --root data/clips --out data/index_clips
[ -f data/index_clips_c40/train.csv ] || $PY src/build_index.py --root data/clips_c40 --out data/index_clips_c40 --compression c40
# 3) điểm mốc MediaPipe, góc quay, khung mắt/mũi/miệng (chạy trên c23, c40 dùng chung)
[ -f data/index_clips/landmarks.npz ] || $PY src/face_landmarks.py --root data/clips --index data/index_clips --points
# 3b) kiểm tra: thiếu video, c40 lệch c23, MediaPipe theo nhóm, ảnh mẫu zoom c23/c40/DFD.
#     Có vấn đề thì dừng ở đây (set -e), không chạy bước 4 tốn nhiều giờ.
$PY src/check_clips.py
# 4) đặc trưng B1 đóng băng, 3 mức zoom + 4 vùng
$PY src/extract_features.py runs/b1_mix/best.pt
# 5) bảng ablation (mỗi dòng vài phút); lọc góc DFD 30° bật mặc định
T="$PY src/temporal.py"
$T --name a1_no_temporal --no-temporal
$T --name a2_temporal
$T --name a3_landmarks --landmarks
$T --name a4_roi --landmarks --roi
$T --name a5_zoom --landmarks --roi --zoom
$T --name a6_dfdtrain --landmarks --roi --zoom --dfd-train
$T --name a6_dfdtrain_noyaw --landmarks --roi --zoom --dfd-train --dfd-max-yaw 999
echo "=== $(date '+%F %T') xong"
