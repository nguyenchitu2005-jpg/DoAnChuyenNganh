"""Chỉ số segmentation vùng giả mạo, tính riêng cho từng phương pháp.

Ảnh giả: IoU, Dice (trung bình theo ảnh), Precision/Recall/F1 mức pixel (cộng dồn
mọi pixel), pixel-AUC (trên một mẫu pixel ngẫu nhiên mỗi ảnh).
Ảnh thật: tỉ lệ báo nhầm = phần trăm ảnh có vùng dự đoán giả lớn hơn `fa_area`
diện tích ảnh (IoU không đo được trên mask đen).
"""
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from dataset import METHODS


class SegMetrics:
    def __init__(self, threshold=0.5, fa_area=0.01, auc_pixels=256, seed=0):
        self.threshold = threshold
        self.fa_area = fa_area
        self.auc_pixels = auc_pixels
        self.generator = torch.Generator().manual_seed(seed)
        self.per_method = defaultdict(lambda: defaultdict(list))

    @torch.no_grad()
    def update(self, probs, masks, labels, methods):
        """probs, masks: (N,1,H,W); labels, methods: (N,) tensor hoặc list."""
        probs = probs.float().flatten(1).cpu()
        gt = masks.flatten(1).cpu() > 0.5
        pred = probs > self.threshold
        inter = (pred & gt).sum(1)
        psum = pred.sum(1)
        gsum = gt.sum(1)
        area = probs.shape[1]
        idx = torch.randint(0, area, (len(probs), self.auc_pixels),
                            generator=self.generator)
        auc_p = probs.gather(1, idx)
        auc_y = gt.gather(1, idx)
        for i in range(len(probs)):
            s = self.per_method[METHODS[int(methods[i])]]
            s['auc_p'].append(auc_p[i].numpy())
            s['auc_y'].append(auc_y[i].numpy())
            if int(labels[i]):
                union = psum[i] + gsum[i] - inter[i]
                s['iou'].append(1.0 if union == 0 else (inter[i] / union).item())
                denom = psum[i] + gsum[i]
                s['dice'].append(1.0 if denom == 0 else
                                 (2 * inter[i] / denom).item())
                s['tp'].append(inter[i].item())
                s['fp'].append((psum[i] - inter[i]).item())
                s['fn'].append((gsum[i] - inter[i]).item())
            else:
                s['area'].append(psum[i].item() / area)

    def _row(self, stats_list):
        merged = defaultdict(list)
        for s in stats_list:
            for k, v in s.items():
                merged[k] += v
        row = {}
        if merged['iou']:
            tp, fp, fn = (sum(merged[k]) for k in ('tp', 'fp', 'fn'))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            row.update(
                n=len(merged['iou']),
                iou=float(np.mean(merged['iou'])),
                dice=float(np.mean(merged['dice'])),
                precision=precision, recall=recall,
                f1=(2 * precision * recall / (precision + recall)
                    if precision + recall else 0.0))
        if merged['area']:
            area = np.array(merged['area'])
            row.update(n_real=len(area),
                       false_alarm=float((area > self.fa_area).mean()),
                       mean_area=float(area.mean()))
        y = np.concatenate(merged['auc_y'])
        if y.any() and not y.all():
            row['auc'] = float(roc_auc_score(y, np.concatenate(merged['auc_p'])))
        return row

    def summary(self):
        """Một dòng cho mỗi phương pháp, cộng 'fake_all' và 'real_all'."""
        out = {m: self._row([s]) for m, s in self.per_method.items()}
        fakes = [s for s in self.per_method.values() if s['iou']]
        reals = [s for s in self.per_method.values() if s['area']]
        if fakes:
            out['fake_all'] = self._row(fakes)
        if reals:
            out['real_all'] = self._row(reals)
        return out


def format_table(summary):
    cols = [('n', '{:>6}'), ('iou', '{:>6.3f}'), ('dice', '{:>6.3f}'),
            ('precision', '{:>6.3f}'), ('recall', '{:>6.3f}'), ('f1', '{:>6.3f}'),
            ('auc', '{:>6.3f}'), ('n_real', '{:>6}'), ('false_alarm', '{:>6.1%}')]
    head = '{:<15}'.format('method') + ''.join(
        '{:>7}'.format(c[:6] if c != 'false_alarm' else 'FA') for c, _ in cols)
    lines = [head]
    order = [m for m in METHODS if m in summary] + [
        m for m in ('fake_all', 'real_all') if m in summary]
    for m in order:
        row = summary[m]
        cells = [(' ' + fmt.format(row[c])) if c in row else '{:>7}'.format('-')
                 for c, fmt in cols]
        lines.append('{:<15}'.format(m) + ''.join(cells))
    return '\n'.join(lines)
