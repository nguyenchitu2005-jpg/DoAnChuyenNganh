"""So 4 cách kết luận video cho một checkpoint: trung bình / trung vị xác suất khung, có / không
lật ngang ảnh khi chấm (TTA). Chạy từ thư mục dự án:
python src/tta_median.py runs/b1_dfd/best.pt
Xác suất khung lưu ở <run>/cls_probs_<tập>[_flip].npz (bản không lật dùng chung với
sweep_cls_threshold.py). Kết quả: <run>/tta_median.csv."""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from evaluate import eval_loader, load_checkpoint

SETS = [('val', 'c23'), ('val', 'c40'), ('test', 'c23'), ('test', 'c40'), ('dfd', 'c23'),
        ('dfd', 'c40'), ('celebdf', 'c23')]


@torch.no_grad()
def frame_probs(ckpt, split, comp, flip):
    model, cfg, _, device = load_checkpoint(ckpt, comp)
    if split == 'celebdf':
        cfg = dict(cfg, data_root='data/clips_celebdf', index_dir='data/index_celebdf')
    loader = eval_loader(cfg, split)
    probs, labels = [], []
    for x, _, label, _ in tqdm(loader, desc=f'{split} {comp} flip={flip}', leave=False):
        if flip:
            x = torch.flip(x, dims=[3])
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            out = model(x.to(device, non_blocking=True), with_cls=True)
        probs.append(torch.sigmoid(out[1].float()).cpu().numpy())
        labels.append(label.numpy())
    videos = np.array([r[3] for r in loader.dataset.rows])
    return np.concatenate(probs), np.concatenate(labels), videos


def load(ckpt, split, comp, flip):
    tag = split if comp == 'c23' else f'{split}_{comp}'
    path = os.path.join(os.path.dirname(ckpt), f'cls_probs_{tag}{"_flip" if flip else ""}.npz')
    if not os.path.isfile(path):
        np.savez(path, **dict(zip('pyv', frame_probs(ckpt, split, comp, flip))))
    d = np.load(path)
    return d['p'], d['y'], d['v']


def video_scores(p, y, v, how):
    g = pd.DataFrame({'p': p, 'y': y, 'v': v}).groupby('v')
    return g.p.agg(how).to_numpy(), g.y.first().to_numpy()


def main():
    ckpt = sys.argv[1]
    rows = []
    for split, comp in SETS:
        p0, y, v = load(ckpt, split, comp, False)
        p1, y1, v1 = load(ckpt, split, comp, True)
        assert (y == y1).all() and (v == v1).all()
        for tta, p in (('không lật', p0), ('lật + gốc', (p0 + p1) / 2)):
            for how, name in (('mean', 'trung bình'), ('median', 'trung vị')):
                s, t = video_scores(p, y, v, how)
                pred = s >= 0.5
                bal = ((~pred[t == 0]).mean() + pred[t == 1].mean()) / 2
                rows.append({'tập': split if comp == 'c23' else f'{split}_{comp}',
                             'cách': f'{name}, {tta}', 'acc cân bằng': round(100 * bal, 1),
                             'AUC': round(roc_auc_score(t, s), 3),
                             'bắt giả': round(100 * pred[t == 1].mean(), 1),
                             'đúng thật': round(100 * (~pred[t == 0]).mean(), 1)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(os.path.dirname(ckpt), 'tta_median.csv'), index=False)
    print(df.pivot(index='cách', columns='tập', values='acc cân bằng').to_string())
    print()
    print(df.pivot(index='cách', columns='tập', values='AUC').to_string())


if __name__ == '__main__':
    main()
