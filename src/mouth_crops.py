"""Cắt vùng miệng từ khung hình gốc (độ phân giải cao) cho mức zoom miệng của extract_features.py.

Ảnh mặt 224 đã thu nhỏ từ video gốc: vùng miệng trong đó chỉ khoảng 60–80 px, phóng lên 224
thì mờ, không thêm chi tiết. Ở đây khung miệng (MediaPipe trên ảnh 224, <index>/landmarks.csv;
khung không thấy mặt dùng trung vị của video) được đổi về toạ độ khung hình gốc qua khung cắt
trong boxes.json, làm vuông, rồi cắt thẳng từ video gốc và đưa về 224.
Bản c40 cắt đúng khung đó trên video c40 (cùng frame, cùng vị trí, chỉ khác mức nén).

Chạy: python src/mouth_crops.py --root data/clips16 --index data/index_clips16 --out data/clips16_mouth
      python src/mouth_crops.py --root data/clips16 --index data/index_clips16 --compression c40 \
          --out data/clips16_mouth_c40
Kết quả: <out>/<nhóm>/frames/<video>/<frame>.png, cùng đường dẫn tương đối với ảnh trong <root>.
Video đã cắt đủ thì bỏ qua (chạy lại an toàn).
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

from preprocess import cut_square


def mouth_square(square, mouth, size):
    """Khung miệng (x1, y1, x2, y2) trên ảnh size×size → khung vuông (x, y, cạnh) trên khung hình gốc."""
    x, y, side = square
    m = np.asarray(mouth, float) * side / size + [x, y, x, y]
    half = max(m[2] - m[0], m[3] - m[1]) / 2
    cx, cy = (m[0] + m[2]) / 2, (m[1] + m[3]) / 2
    return int(round(cx - half)), int(round(cy - half)), max(1, int(round(2 * half)))


def work(task):
    src, out_dir, crops, size = task
    os.makedirs(out_dir, exist_ok=True)
    wanted = {int(n[:-4]): n for n in crops}
    cap = cv2.VideoCapture(src)
    sides = []
    for i in range(max(wanted) + 1):
        if not cap.grab():
            break
        if i in wanted:
            ok, f = cap.retrieve()
            if ok:
                sq = crops[wanted[i]]
                cv2.imwrite(os.path.join(out_dir, wanted[i]), cut_square(f, sq, size, None),
                            [cv2.IMWRITE_PNG_COMPRESSION, 3])
                sides.append(sq[2])
    cap.release()
    return os.path.basename(os.path.dirname(os.path.dirname(out_dir))), sides   # tên nhóm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/clips16', help='bản c23 đã cắt (có boxes.json)')
    ap.add_argument('--index', default='data/index_clips16', help='chứa landmarks.csv')
    ap.add_argument('--src', default='data/FaceForensics')
    ap.add_argument('--compression', default='c23', choices=['c23', 'c40'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    from extract_features import REGIONS, region_boxes   # nạp torch, chỉ tiến trình chính cần
    boxes_of, _ = region_boxes(os.path.join(args.index, 'landmarks.csv'))
    k = REGIONS.index('mouth')

    tasks = []
    for bj in sorted(glob.glob(os.path.join(args.root, '*', '*', 'frames', '*', 'boxes.json'))):
        rel = os.path.relpath(os.path.dirname(bj), args.root).replace('\\', '/')   # nhóm/frames/video
        group, video = rel.rsplit('/frames/', 1)
        with open(bj, encoding='utf-8') as f:
            squares = json.load(f)
        crops = {n: mouth_square(sq, boxes_of[f'{rel}/{n}'][k], args.size)
                 for n, sq in squares.items() if f'{rel}/{n}' in boxes_of}
        out_dir = os.path.join(args.out, rel)
        if not crops or all(os.path.isfile(os.path.join(out_dir, n)) for n in crops):
            continue
        src = os.path.join(args.src, group, args.compression, 'videos', video + '.mp4')
        tasks.append((src, out_dir, crops, args.size))
    print(f'{len(tasks)} video cần cắt vùng miệng')
    sides = defaultdict(list)
    with Pool(args.workers) as pool:
        for group, s in tqdm(pool.imap_unordered(work, tasks), total=len(tasks)):
            sides[group] += s
    for g, s in sides.items():   # cạnh vùng miệng trên khung hình gốc: > size thì có thêm chi tiết
        s = np.array(s)
        print(f'{g:<20} cạnh vùng miệng gốc: trung vị {np.median(s):.0f} px, '
              f'> {args.size} px: {(s > args.size).mean():.0%}, < 112 px: {(s < 112).mean():.0%}')


if __name__ == '__main__':
    main()
