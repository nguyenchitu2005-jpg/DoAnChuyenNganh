"""Điểm mốc MediaPipe FaceLandmarker trên mọi ảnh mặt đã cắt: góc quay đầu và khung mắt, mũi, miệng
(cho lọc khung chính diện và gộp đặc trưng theo vùng), tuỳ chọn lưu điểm mốc chọn lọc.

Cần: pip install mediapipe --no-deps (tránh kéo opencv-contrib ghi đè opencv-python) và file
mô hình data/models/face_landmarker.task:
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
Chạy: python src/face_landmarks.py                     → data/index/landmarks.csv
      python src/face_landmarks.py --root data/clips --index data/index_clips --points
Mỗi dòng: frame, yaw, pitch, roll (độ; yaw = quay trái/phải, 0 = nhìn thẳng), rồi 4 khung vuông
x1,y1,x2,y2 theo pixel ảnh 224: miệng, mắt trái, mắt phải, mũi. Ảnh không thấy mặt: các ô trống.
--points: thêm <out>.npz, POINTS điểm (x, y, z chuẩn hoá 0-1) mỗi ảnh, NaN nếu không thấy mặt.
Ảnh c40 cùng frame, cùng khung cắt với c23 (preprocess.py --boxes-from) nên chỉ chạy trên c23.
Chạy lại an toàn: frame đã có trong file csv thì bỏ qua (riêng --points luôn chạy lại hết).
"""
import argparse
import csv
import math
import os
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

# chỉ số điểm mốc của MediaPipe Face Mesh (FACEMESH_LIPS, _LEFT_EYE, _RIGHT_EYE, mũi)
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
NOSE = [1, 2, 4, 5, 6, 48, 49, 64, 98, 168, 195, 197, 278, 279, 294, 327]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
             377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
BROWS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
POINTS = LIPS + LEFT_EYE + RIGHT_EYE + NOSE + FACE_OVAL + BROWS          # 114 điểm
REGIONS = {'mouth': (LIPS, 1.8), 'leye': (LEFT_EYE, 1.6), 'reye': (RIGHT_EYE, 1.6),
           'nose': (NOSE, 1.3)}    # (điểm, hệ số nới khung); miệng lấy cả quanh môi và cằm trên
MODEL = 'data/models/face_landmarker.task'
_lm = None


def init():
    global _lm
    from mediapipe.tasks.python import BaseOptions, vision
    _lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL), num_faces=1,
        output_facial_transformation_matrixes=True))


def euler(m):
    """yaw, pitch, roll (độ) từ ma trận xoay 3×3 của MediaPipe."""
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -m[2][0]))))
    pitch = math.degrees(math.atan2(m[2][1], m[2][2]))
    roll = math.degrees(math.atan2(m[1][0], m[0][0]))
    return yaw, pitch, roll


def square_box(pts, expand):
    (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
    cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * expand
    return [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]


def analyse(img_rgb):
    """(hàng csv không kèm frame, mảng POINTS×3) hoặc (None, None) nếu không thấy mặt."""
    import mediapipe as mp
    r = _lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb)))
    if not r.face_landmarks:
        return None, None
    h, w = img_rgb.shape[:2]
    lms = np.array([(p.x, p.y, p.z) for p in r.face_landmarks[0]], np.float32)
    row = list(euler(np.array(r.facial_transformation_matrixes[0])[:3, :3]))
    for idx, expand in REGIONS.values():
        row += square_box(lms[idx, :2] * (w, h), expand)
    return [round(v, 1) for v in row], lms[POINTS]


def work(args):
    root, frame = args
    img = cv2.imread(os.path.join(root, frame))
    if img is None:
        return frame, None, None
    return (frame,) + analyse(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/faces')
    ap.add_argument('--index', default='data/index')
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test', 'dfd'])
    ap.add_argument('--out', help='mặc định <index>/landmarks.csv')
    ap.add_argument('--points', action='store_true', help='lưu thêm điểm mốc chọn lọc (.npz)')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument('--limit', type=int, help='chỉ chạy N ảnh đầu (thử)')
    args = ap.parse_args()
    out = args.out or os.path.join(args.index, 'landmarks.csv')
    done = set()
    if os.path.isfile(out) and not args.points:
        with open(out, encoding='utf-8') as f:
            done = {r['frame'] for r in csv.DictReader(f)}
    frames = []
    for split in args.splits:
        with open(os.path.join(args.index, split + '.csv'), encoding='utf-8') as f:
            frames += [r['frame'] for r in csv.DictReader(f) if r['frame'] not in done]
    frames = sorted(set(frames))[:args.limit]
    head = ['frame', 'yaw', 'pitch', 'roll'] + [f'{k}_{c}' for k in REGIONS
                                                for c in ('x1', 'y1', 'x2', 'y2')]
    pts = np.full((len(frames), len(POINTS), 3), np.nan, np.float32) if args.points else None
    pos = {f: i for i, f in enumerate(frames)}
    mode = 'w' if args.points or not os.path.isfile(out) else 'a'
    miss = 0
    with open(out, mode, newline='', encoding='utf-8') as f, Pool(args.workers, init) as pool:
        w = csv.writer(f)
        if mode == 'w':
            w.writerow(head)
        for frame, row, p in tqdm(pool.imap_unordered(work, [(args.root, fr) for fr in frames], 64),
                                  total=len(frames)):
            miss += row is None
            w.writerow([frame] + (row or [''] * (len(head) - 1)))
            if pts is not None and p is not None:
                pts[pos[frame]] = p
    if pts is not None:
        np.savez(os.path.splitext(out)[0] + '.npz', frames=np.array(frames), points=pts)
    print(f'{len(frames)} ảnh, {miss} ảnh không thấy mặt → {out}')


if __name__ == '__main__':
    main()
