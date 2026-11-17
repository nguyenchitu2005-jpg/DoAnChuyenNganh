"""Dò ngưỡng pixel trên tập val: chạy mô hình một lần, tính IoU/báo nhầm cho nhiều ngưỡng.

Chạy: python src/sweep_threshold.py runs/b1_mix/best.pt
      (mặc định val c23 và c40; --compression c23 để chỉ dò c23)
Chọn ngưỡng có điểm (IoU ảnh giả + 1 - báo nhầm)/2 trung bình các mức nén cao nhất,
giống tiêu chí chọn checkpoint lúc train. Kết quả: <thư mục run>/threshold_sweep.json
Chỉ dò trên val; đo lại trên test bằng ngưỡng đã chọn để không "nhìn trộm" test.
"""
import argparse
import json
import os

import torch
from tqdm import tqdm

from evaluate import eval_loader, load_checkpoint
from metrics import SegMetrics

THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
SHOW = ['Face2Face', 'NeuralTextures', 'fake_all']


@torch.no_grad()
def sweep(model, loader, device):
    metrics = {t: SegMetrics(threshold=t) for t in THRESHOLDS}
    for x, y, label, method in tqdm(loader, leave=False):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            out = model(x.to(device, non_blocking=True))
        probs = torch.sigmoid((out[0] if isinstance(out, tuple) else out).float())
        for m in metrics.values():
            m.update(probs, y, label, method)
    return {t: m.summary() for t, m in metrics.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--compression', nargs='+', default=['c23', 'c40'])
    p.add_argument('--fraction', type=float, default=1.0)
    args = p.parse_args()

    res = {}
    for comp in args.compression:
        model, cfg, ckpt, device = load_checkpoint(args.checkpoint, comp)
        res[comp] = sweep(model, eval_loader(cfg, 'val', args.fraction), device)
        print(f'\n{cfg["name"]} (epoch {ckpt["epoch"]}), val {comp}')
        print('ngưỡng ' + ''.join(f'{"IoU " + m[:6]:>13}' for m in SHOW) + f'{"báo nhầm":>10}{"điểm":>8}')
        for t, s in res[comp].items():
            fa = s['real_all']['false_alarm']
            score = (s['fake_all']['iou'] + 1 - fa) / 2
            print(f'{t:>6.2f} ' + ''.join(f'{s[m]["iou"]:>13.3f}' for m in SHOW)
                  + f'{fa:>10.1%}{score:>8.3f}')

    def score(t):
        return sum((res[c][t]['fake_all']['iou'] + 1 - res[c][t]['real_all']['false_alarm']) / 2
                   for c in res) / len(res)
    best = max(THRESHOLDS, key=score)
    assert best in THRESHOLDS and 0.5 in THRESHOLDS   # luôn so được với ngưỡng mặc định
    print(f'\nNgưỡng tốt nhất trên val ({"+".join(res)}): {best} '
          f'(điểm {score(best):.3f}, ngưỡng 0.5: {score(0.5):.3f})')
    out = os.path.join(os.path.dirname(args.checkpoint), 'threshold_sweep.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'best': best, 'results': res}, f, indent=1)
    print('saved', out)


if __name__ == '__main__':
    main()
