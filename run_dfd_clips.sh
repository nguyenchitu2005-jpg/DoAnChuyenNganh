#!/usr/bin/env bash
# Cắt lại DFD (FF++ giữ nguyên data/faces, data/clips): 4 đoạn × 8 khung liên tiếp mỗi video,
# đoạn đặt ở chỗ mặt chính diện gần các mốc giữa video (không sát đầu/cuối), bỏ khung không thấy
# mặt, khung cắt bám theo mặt và được dịch vào trong video (không dải đen).
# Dùng cho cả B1 (từng khung) lẫn mô hình thời gian. Không dùng Kaggle.
# Chạy: bash run_dfd_clips.sh   (log: data/clips_dfd.log; bước đã xong tự bỏ qua)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
D=data/clips_dfd; I=data/index_clips_dfd; G="--groups actors DeepFakeDetection"
exec > >(tee -a data/clips_dfd.log) 2>&1
echo "=== $(date '+%F %T') bắt đầu"
$PY src/preprocess.py --out $D --clip-len 8 --follow-face --frontal 0.3 $G || true   # lỗi GPU: dòng sau chạy bù
$PY src/preprocess.py --out $D --clip-len 8 --follow-face --frontal 0.3 $G
$PY src/preprocess.py --compression c40 --out ${D}_c40 --boxes-from $D --workers 4 $G
[ -f $I/dfd.csv ] || $PY src/build_index.py --root $D --out $I
[ -f ${I}_c40/dfd.csv ] || $PY src/build_index.py --root ${D}_c40 --out ${I}_c40 --compression c40
[ -f $I/landmarks.npz ] || $PY src/face_landmarks.py --root $D --index $I --splits dfd --points
echo "=== $(date '+%F %T') xong"
