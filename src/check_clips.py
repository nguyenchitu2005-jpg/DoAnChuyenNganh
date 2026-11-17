"""Kiểm tra dữ liệu đoạn khung trước khi trích đặc trưng (run_temporal.sh gọi giữa bước 3 và 4).

Chạy: python src/check_clips.py [--clips data/clips16 --index data/index_clips16]
1) Mọi video gốc c23 đều đã cắt (có boxes.json), số đoạn / khung mỗi video.
2) c40 khớp từng khung với c23 (cùng đường dẫn, cùng thứ tự trong index).
3) MediaPipe: tỉ lệ khung không thấy mặt theo nhóm (khung đó dùng khung vùng trung vị).
4) Ảnh mẫu 3 mức zoom cho c23, c40 và DFD → runs/temporal/check_zoom_<thư mục>_<nhóm>.jpg
Thoát với mã 1 nếu thiếu > 1% video hoặc c40 lệch c23, để run_temporal.sh dừng trước bước
trích đặc trưng tốn nhiều giờ.
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
from torchvision.ops import roi_align

sys.path.insert(0, os.path.dirname(__file__))
from extract_features import REGIONS, region_boxes   # noqa: E402

SRC = 'data/FaceForensics'
OUT = 'runs/temporal'
GROUPS = ['original_sequences/youtube', 'original_sequences/actors'] + [
    'manipulated_sequences/' + m for m in
    ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'DeepFakeDetection']]


def read_index(path):
    rows = []
    for sp in ('train', 'val', 'test', 'dfd'):
        with open(os.path.join(path, sp + '.csv'), encoding='utf-8') as f:
            rows += [dict(r, split=sp) for r in csv.DictReader(f)]
    return rows


def zoom_sheet(rows, root, boxes_of, out, n=6):
    """Mỗi hàng: mức 1 | mức 2 (giữa 172 px) | mức 3 (miệng), giống extract_features.py."""
    pick = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, n)]
    lines = []
    for r in pick:
        img = cv2.imread(os.path.join(root, r['frame']))
        size = img.shape[0]
        c = (size - size / 1.3) / 2
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float()
        boxes = torch.tensor([[c, c, size - c, size - c],
                              boxes_of[r['frame']][REGIONS.index('mouth')].tolist()])
        crops = roi_align(x, [boxes], output_size=size, spatial_scale=1.0, sampling_ratio=2,
                          aligned=True)
        z = [img] + [cr.permute(1, 2, 0).clamp(0, 255).byte().numpy() for cr in crops]
        lines.append(np.hstack(z))
    cv2.imwrite(out, np.vstack(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', default='data/clips')
    ap.add_argument('--index', default='data/index_clips')
    args = ap.parse_args()
    CLIPS, CLIPS40 = args.clips, args.clips + '_c40'
    IDX, IDX40 = args.index, args.index + '_c40'
    tag = os.path.basename(CLIPS)
    bad = False
    # 1) video đã cắt
    print('1) video c23 đã cắt (có boxes.json) / video gốc; đoạn mỗi video')
    for g in GROUPS:
        vids = sorted(os.path.splitext(os.path.basename(v))[0]
                      for v in glob.glob(f'{SRC}/{g}/c23/videos/*.mp4'))
        done, clips = 0, Counter()
        for v in vids:
            p = f'{CLIPS}/{g}/frames/{v}/boxes.json'
            if os.path.isfile(p):
                done += 1
                with open(p, encoding='utf-8') as f:
                    idx = sorted(int(k[:-4]) for k in json.load(f))
                clips[sum(1 for a, b in zip([None] + idx, idx) if a is None or b - a > 8)] += 1   # đoạn mới = cách > 8 frame
        miss = len(vids) - done
        bad |= miss > 0.01 * len(vids)
        print(f'   {g:<42} {done}/{len(vids)}  thiếu {miss}  số đoạn: {dict(sorted(clips.items()))}')

    # 2) c40 khớp c23
    a, b = read_index(IDX), read_index(IDX40)
    same = [r['frame'] for r in a] == [r['frame'] for r in b]
    lost = sum(not os.path.isfile(os.path.join(CLIPS40, r['frame'])) for r in a)
    bad |= not same or lost > 0
    print(f'2) index c23 {len(a)} khung, c40 {len(b)} khung, cùng thứ tự: {same}, '
          f'ảnh c40 thiếu: {lost}')

    # 3) MediaPipe
    with open(os.path.join(IDX, 'landmarks.csv'), encoding='utf-8') as f:
        lm = {r['frame']: r for r in csv.DictReader(f)}
    stat = defaultdict(lambda: [0, 0, 0])
    for r in a:
        s = stat[r['method']]
        s[0] += 1
        s[1] += not lm.get(r['frame'], {}).get('yaw')
        y = lm.get(r['frame'], {}).get('yaw')
        s[2] += bool(y) and abs(float(y)) >= 30
    print('3) MediaPipe: khung không thấy mặt / |yaw| ≥ 30° theo nhóm')
    for k, (n, m, side) in stat.items():
        print(f'   {k:<16} {n:>7} khung  không thấy mặt {m / n:6.1%}  nghiêng ≥30° {side / n:6.1%}')

    # 4) ảnh mẫu zoom
    boxes_of, _ = region_boxes(os.path.join(IDX, 'landmarks.csv'))
    os.makedirs(OUT, exist_ok=True)
    for name, root, rows in [('c23', CLIPS, [r for r in a if r['split'] == 'test']),
                             ('c40', CLIPS40, [r for r in b if r['split'] == 'test']),
                             ('dfd', CLIPS, [r for r in a if r['split'] == 'dfd'])]:
        zoom_sheet(rows, root, boxes_of, f'{OUT}/check_zoom_{tag}_{name}.jpg')
    print(f'4) ảnh mẫu zoom: {OUT}/check_zoom_{tag}_{{c23,c40,dfd}}.jpg')
    print('KẾT QUẢ:', 'CÓ VẤN ĐỀ, dừng' if bad else 'ổn')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
