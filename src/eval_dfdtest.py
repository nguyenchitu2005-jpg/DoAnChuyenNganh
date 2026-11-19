"""Chấm một checkpoint train không có DFD (ảnh data/faces) trên đúng DFD test của phương án B
(14 diễn viên test, temporal.dfd_part), để so sánh công bằng với b1_dfd.

Chạy: python src/eval_dfdtest.py runs/b1_mix/best.pt [c23|c40]
Ghi data/index_dfdtest[_c40]/dfd.csv và <thư mục run>/eval_dfdtest[_c40].json
"""
import json
import os
import sys

import pandas as pd
from evaluate import load_checkpoint, run_eval, eval_loader
from temporal import dfd_part


def main():
    ckpt_path, comp = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'c23')
    model, cfg, ckpt, device = load_checkpoint(ckpt_path, comp)
    out_idx = 'data/index_dfdtest' + ('' if comp == 'c23' else '_' + comp)
    os.makedirs(out_idx, exist_ok=True)
    d = pd.read_csv(os.path.join(cfg['index_dir'], 'dfd.csv'), dtype=str, keep_default_na=False)
    d = d[[dfd_part(v) == 'test' for v in d.video]]
    d.to_csv(os.path.join(out_idx, 'dfd.csv'), index=False)
    print(len(d), 'khung', d.video.nunique(), 'video')
    cfg = dict(cfg, index_dir=out_idx)
    summary = run_eval(model, eval_loader(cfg, 'dfd'), device, cfg['threshold'], desc='dfdtest')
    tag = 'dfdtest' + ('' if comp == 'c23' else '_' + comp)
    json.dump({'epoch': ckpt['epoch'], 'split': 'dfdtest', 'compression': comp, 'cascade': False,
               'fraction': 1.0, 'metrics': summary},
              open(os.path.join(os.path.dirname(ckpt_path), f'eval_{tag}.json'), 'w', encoding='utf-8'), indent=2)
    v = summary['cls']['video']
    print(tag, 'bal', round((v['bin_recall'] + v['bin_real_recall']) / 2 * 100, 1), 'auc', round(v['bin_auc'], 3))


if __name__ == '__main__':
    main()
