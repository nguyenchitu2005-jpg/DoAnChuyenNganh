#!/usr/bin/env bash
# Đoạn dài cho mô hình thời gian (+ vùng miệng cắt từ khung hình gốc): 2 đoạn × 16 frame, cách 1 frame lấy 1 (≈ 1,1–1,3 s, đủ một lần
# chớp mắt và một chu kỳ miệng; đoạn 8 frame liên tiếp cũ chỉ ≈ 0,3 s). Bỏ frame không thấy mặt,
# khung cắt bám theo mặt khi người di chuyển. Thêm bản DFD chỉ chọn đoạn mặt chính diện.
# So sánh 3 bản (cũ 8 frame / 16 frame / 16 frame chính diện), mỗi bản 3 seed. Không dùng Kaggle.
# Chạy: bash run_clips16.sh   (log: runs/temporal/clips16.log; bước đã xong tự bỏ qua)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
CLIP="--clip-len 16 --clip-stride 2"
mkdir -p runs/temporal
exec > >(tee -a runs/temporal/clips16.log) 2>&1
echo "=== $(date '+%F %T') bắt đầu"

# 1) FF++ + DFD, đoạn 16 frame
D=data/clips16; I=data/index_clips16; F=runs/b1_mix/features16
$PY src/preprocess.py --out $D $CLIP --follow-face || true   # lỗi GPU giữa chừng: dòng sau chạy bù
$PY src/preprocess.py --out $D $CLIP --follow-face     # bù video lỗi giữa chừng (nếu có)
$PY src/preprocess.py --compression c40 --out ${D}_c40 --boxes-from $D --workers 4
[ -f $I/dfd.csv ] || $PY src/build_index.py --root $D --out $I
[ -f ${I}_c40/dfd.csv ] || $PY src/build_index.py --root ${D}_c40 --out ${I}_c40 --compression c40
[ -f $I/landmarks.npz ] || $PY src/face_landmarks.py --root $D --index $I --points
$PY src/check_clips.py --clips $D --index $I
# vùng miệng cắt thẳng từ khung hình gốc (mức zoom 4, độ phân giải cao hơn mức 3 phóng từ ảnh 224).
# Ảnh miệng chỉ là bước trung gian (~17 GB mỗi mức nén): cắt → trích đặc trưng → xoá, từng mức nén.
mouth_features() {   # $1 root, $2 index, $3 thư mục đặc trưng, $4 c23|c40, $5 split cuối, các split còn lại
  local sfx=""; [ $4 = c40 ] && sfx=_c40
  [ -f $3/$5$sfx.npz ] && return 0
  $PY src/mouth_crops.py --root $1 --index $2 --compression $4 --out $1_mouth$sfx
  $PY src/extract_features.py runs/b1_mix/best.pt --root $1 --index $2 --out-dir $3 \
      --compression $4 --mouth-root $1_mouth --splits "${@:6}" $5
  rm -rf $1_mouth$sfx
}
mouth_features $D $I $F c23 dfd train val test
mouth_features $D $I $F c40 dfd train val test

# 2) bản DFD chính diện: chỉ cắt lại DFD; FF++ dùng chung index, điểm mốc, đặc trưng của bước 1
G="--groups actors DeepFakeDetection"
DF=data/clips16_frontal; IF=data/index_clips16_frontal; FF=runs/b1_mix/features16_frontal
$PY src/preprocess.py --out $DF $CLIP --follow-face --frontal 0.3 $G
$PY src/preprocess.py --compression c40 --out ${DF}_c40 --boxes-from $DF --workers 4 $G
[ -f $IF/dfd.csv ] || $PY src/build_index.py --root $DF --out $IF
[ -f ${IF}_c40/dfd.csv ] || $PY src/build_index.py --root ${DF}_c40 --out ${IF}_c40 --compression c40
[ -f $IF/landmarks_dfd.npz ] || $PY src/face_landmarks.py --root $DF --index $IF --splits dfd --points \
    --out $IF/landmarks_dfd.csv
$PY - "$I" "$F" "$IF" "$FF" <<'EOF'
import csv, os, shutil, sys
import numpy as np
I, F, IF, FF = sys.argv[1:]
os.makedirs(FF, exist_ok=True)
for sfx in ('', '_c40'):
    for s in ('train', 'val', 'test'):
        shutil.copy(f'{I}{sfx}/{s}.csv', f'{IF}{sfx}/{s}.csv')
        if not os.path.exists(f'{FF}/{s}{sfx}.npz'):
            os.link(f'{F}/{s}{sfx}.npz', f'{FF}/{s}{sfx}.npz')
is_dfd = lambda f: f.startswith(('original_sequences/actors', 'manipulated_sequences/DeepFakeDetection'))
with open(f'{I}/landmarks.csv', encoding='utf-8') as f:
    old = [r for r in csv.DictReader(f) if not is_dfd(r['frame'])]
with open(f'{IF}/landmarks_dfd.csv', encoding='utf-8') as f:
    rd = csv.DictReader(f); new = list(rd); fields = rd.fieldnames
with open(f'{IF}/landmarks.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(old + new)
a, b = np.load(f'{I}/landmarks.npz'), np.load(f'{IF}/landmarks_dfd.npz')
keep = np.array([not is_dfd(str(x)) for x in a['frames']])
np.savez(f'{IF}/landmarks.npz', frames=np.concatenate([a['frames'][keep], b['frames']]),
         points=np.concatenate([a['points'][keep], b['points']]))
print(f'{IF}: điểm mốc FF++ {len(old)} + DFD {len(new)}')
EOF
mouth_features $DF $IF $FF c23 dfd
mouth_features $DF $IF $FF c40 dfd

# 3) so sánh, cấu hình a6 (DFD train, lọc khung xấu, không lọc góc), 3 seed mỗi bản.
#    Bản cũ 8 frame đã chạy: runs/temporal/check_grouping{,_s1,_s2}
#    16 frame: không zoom / zoom 3 mức / zoom với miệng từ khung gốc; bản chính diện dùng miệng gốc
A="--landmarks --roi --dfd-train --dfd-max-bad 0.5 --dfd-max-yaw 999 $CLIP"
for s in 0 1 2; do
  $PY src/temporal.py --name c16_nozoom_s$s --seed $s $A --features $F --index $I
  $PY src/temporal.py --name c16_zoom_s$s --seed $s $A --zoom --features $F --index $I
  $PY src/temporal.py --name c16_hires_s$s --seed $s $A --zoom-hires --features $F --index $I
  $PY src/temporal.py --name c16_frontal_hires_s$s --seed $s $A --zoom-hires --features $FF --index $IF
done
echo "=== $(date '+%F %T') xong"
