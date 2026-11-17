#!/usr/bin/env bash
# Cắt lại FF++ cho mô hình thời gian: 4 đoạn × 8 khung
# liên tiếp mỗi video, đoạn cách đều giữa video (không chọn đoạn chính diện như DFD), bỏ khung không thấy mặt,
# khung cắt bám mặt và dịch vào trong video. data/clips (bản cũ) và data/faces giữ nguyên.
# Chạy: bash run_ff_clips.sh   (log: data/clips_ff.log; bước đã xong tự bỏ qua)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
D=data/clips_ff; I=data/index_clips_ff; G="--groups youtube Deepfakes Face2Face FaceSwap NeuralTextures"
exec > >(tee -a data/clips_ff.log) 2>&1
echo "=== $(date '+%F %T') bắt đầu"
$PY src/preprocess.py --out $D --clip-len 8 --follow-face $G || true   # lỗi GPU: dòng sau chạy bù
$PY src/preprocess.py --out $D --clip-len 8 --follow-face $G
$PY src/preprocess.py --compression c40 --out ${D}_c40 --boxes-from $D --workers 4 $G
[ -f $I/test.csv ] || $PY src/build_index.py --root $D --out $I
[ -f ${I}_c40/test.csv ] || $PY src/build_index.py --root ${D}_c40 --out ${I}_c40 --compression c40
[ -f $I/landmarks.npz ] || $PY src/face_landmarks.py --root $D --index $I --splits train val test --points
echo "=== $(date '+%F %T') xong"
