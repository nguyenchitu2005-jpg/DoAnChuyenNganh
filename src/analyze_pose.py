"""Phân loại thật/giả theo góc quay mặt: khung nghiêng có kém hơn không, và lọc khung nghiêng
khi kết luận cả video có giúp không. Không chạy mô hình: dùng xác suất từng khung đã lưu
(src/sweep_cls_threshold.py → <run>/cls_probs_*.npz) và góc quay (src/face_landmarks.py).

Chạy: python src/analyze_pose.py runs/b1_mix
1) Mức khung, theo nhóm |yaw|: số khung, acc cân bằng (ngưỡng 0,5), AUC.
2) Mức video: chỉ lấy trung bình các khung có |yaw| < θ (khung MediaPipe không thấy mặt bị
   loại); video không còn khung nào thì dùng mọi khung. θ chọn trên val (trung bình c23, c40),
   rồi đo trên test và DFD. Kết quả: <run>/pose_analysis.json
"""
import argparse
import csv
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

SETS = {'val': 'val', 'val_c40': 'val', 'test': 'test', 'test_c40': 'test', 'dfd': 'dfd'}
BINS = [(0, 20), (20, 45), (45, 181)]
THETAS = [15, 20, 25, 30, 40, 45, 60, 181]   # 181 = không lọc


def bal_acc(p, y, t=0.5):
    pred = p > t
    return ((~pred[y == 0]).mean() + pred[y == 1].mean()) / 2


def auc(p, y):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else float('nan')


def video_score(p, y, v, keep):
    _, first, inv = np.unique(v, return_index=True, return_inverse=True)
    n_keep = np.bincount(inv, keep)
    num = np.where(n_keep[inv] > 0, p * keep, p)          # video không còn khung: dùng mọi khung
    den = np.where(n_keep > 0, n_keep, np.bincount(inv))
    return np.bincount(inv, num) / den, y[first], (n_keep > 0).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--landmarks', default='data/index/landmarks.csv')
    args = ap.parse_args()
    with open(args.landmarks, encoding='utf-8') as f:
        yaw = {r['frame']: abs(float(r['yaw'])) if r['yaw'] else np.nan for r in csv.DictReader(f)}

    data = {}
    for tag, split in SETS.items():
        d = np.load(os.path.join(args.run_dir, f'cls_probs_{tag}.npz'))
        with open(f'data/index/{split}.csv', encoding='utf-8') as f:   # c40 cùng thứ tự khung
            frames = [r['frame'] for r in csv.DictReader(f)]
        assert len(frames) == len(d['p']), (tag, len(frames), len(d['p']))
        data[tag] = (d['p'], d['y'], d['v'], np.array([yaw.get(fr, np.nan) for fr in frames]))

    out = {'frame': {}, 'video': {}}
    print('1) Mức khung theo |yaw| (acc cân bằng @0,5 / AUC; số khung)')
    print(' ' * 9 + ' | '.join(f'{h:^28}' for h in ('<20°', '20–45°', '>45°', 'không thấy mặt')))
    for tag, (p, y, v, a) in data.items():
        cells = []
        for lo, hi in BINS + [(None, None)]:
            m = np.isnan(a) if lo is None else (a >= lo) & (a < hi)
            r = (int(m.sum()), bal_acc(p[m], y[m]), auc(p[m], y[m])) if m.sum() else (0, np.nan, np.nan)
            out['frame'].setdefault(tag, {})['không thấy mặt' if lo is None else f'{lo}-{hi}'] = r
            cells.append(f'{r[1]:6.1%} / {r[2]:.3f} ({r[0]:>6})')
        print(f'{tag:<9}' + ' | '.join(cells))

    print('\n2) Mức video, chỉ dùng khung |yaw| < θ (acc cân bằng / AUC; % video còn khung)')
    for th in THETAS:
        row = {}
        for tag, (p, y, v, a) in data.items():
            keep = np.ones_like(p) if th == 181 else (np.nan_to_num(a, nan=999) < th).astype(float)
            vp, vy, frac = video_score(p, y, v, keep)
            row[tag] = (bal_acc(vp, vy), auc(vp, vy), frac)
        out['video'][th] = row
        name = 'không lọc' if th == 181 else f'θ = {th}°'
        print(f'{name:<10}' + '  '.join(f'{t}: {r[0]:.1%}/{r[1]:.3f} ({r[2]:.0%})' for t, r in row.items()))
    best = max(THETAS, key=lambda t: out['video'][t]['val'][0] + out['video'][t]['val_c40'][0])
    print(f'\nθ chọn trên val: {"không lọc" if best == 181 else best}')
    out['best_theta'] = best
    with open(os.path.join(args.run_dir, 'pose_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == '__main__':
    main()
