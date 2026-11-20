"""Dò ngưỡng kết luận thật/giả mức video (trung bình xác suất giả các frame) cho mô hình đa nhiệm.

Chạy: python src/sweep_cls_threshold.py runs/b1_mix/best.pt [runs/c1_mix/best.pt ...]
Chọn ngưỡng có accuracy cân bằng trung bình (val c23, val c40) cao nhất, rồi đo ngưỡng đó
trên test c23, test c40, DFD. Cột "tốt nhất (tham khảo)" chọn ngưỡng ngay trên chính tập
đó: chỉ để biết trần, không dùng được thật vì phải nhìn nhãn tập đó.
Xác suất từng frame lưu ở <run>/cls_probs_<tập>.npz: chạy lại không phải chạy mô hình.
Kết quả: <run>/cls_threshold_sweep.json
"""
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from evaluate import eval_loader, load_checkpoint

SETS = [('val', 'c23'), ('val', 'c40'), ('test', 'c23'), ('test', 'c40'), ('dfd', 'c23'),
        ('dfd', 'c40')]
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)


@torch.no_grad()
def frame_probs(ckpt_path, split, comp):
    model, cfg, _, device = load_checkpoint(ckpt_path, comp)
    loader = eval_loader(cfg, split)
    probs, labels = [], []
    for x, _, label, _ in tqdm(loader, desc=f'{split} {comp}', leave=False):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            out = model(x.to(device, non_blocking=True), with_cls=True)
        probs.append(torch.sigmoid(out[1].float()).cpu().numpy())
        labels.append(label.numpy())
    videos = np.array([r[3] for r in loader.dataset.rows])   # loader không xáo trộn
    return np.concatenate(probs), np.concatenate(labels), videos


def video_level(p, y, v):
    _, first, inv = np.unique(v, return_index=True, return_inverse=True)
    return np.bincount(inv, p) / np.bincount(inv), y[first]


def balanced_acc(p, y, t):
    pred = p > t
    return ((~pred[y == 0]).mean() + pred[y == 1].mean()) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoints', nargs='+')
    for ckpt in ap.parse_args().checkpoints:
        run_dir = os.path.dirname(ckpt)
        vid = {}
        for split, comp in SETS:
            tag = split if comp == 'c23' else f'{split}_{comp}'
            path = os.path.join(run_dir, f'cls_probs_{tag}.npz')
            if not os.path.isfile(path):
                np.savez(path, **dict(zip('pyv', frame_probs(ckpt, split, comp))))
            d = np.load(path)
            vid[tag] = video_level(d['p'], d['y'], d['v'])

        curve = {tag: [balanced_acc(p, y, t) for t in THRESHOLDS] for tag, (p, y) in vid.items()}
        val_mean = (np.array(curve['val']) + np.array(curve['val_c40'])) / 2
        best = float(THRESHOLDS[int(val_mean.argmax())])
        i05, ib = list(THRESHOLDS).index(0.5), list(THRESHOLDS).index(best)
        print(f'\n{run_dir}: ngưỡng chọn trên val = {best}')
        print(f'{"tập":<10}{"ngưỡng 0.5":>12}{"ngưỡng " + str(best):>14}{"tốt nhất (tham khảo)":>26}')
        for tag, c in curve.items():
            k = int(np.argmax(c))
            print(f'{tag:<10}{c[i05]:>12.1%}{c[ib]:>14.1%}{c[k]:>17.1%} @ {THRESHOLDS[k]}')
        with open(os.path.join(run_dir, 'cls_threshold_sweep.json'), 'w', encoding='utf-8') as f:
            json.dump({'best_val': best, 'thresholds': THRESHOLDS.tolist(),
                       'balanced_acc': curve}, f, indent=1)


if __name__ == '__main__':
    main()
