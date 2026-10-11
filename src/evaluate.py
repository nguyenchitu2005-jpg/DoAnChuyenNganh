"""Đánh giá một checkpoint trên val, test (FF++) hoặc dfd (kiểm tra chéo).

Chạy: python src/evaluate.py runs/unet_effb0/best.pt --split test
Độ bền: python src/evaluate.py runs/unet_effb0/best.pt --split test --compression c40
Kết quả in ra bảng theo từng phương pháp và lưu vào <thư mục run>/eval_<split>[_c40].json
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FaceSegDataset, eval_transform
from metrics import SegMetrics, format_table
from models import build_model


@torch.no_grad()
def run_eval(model, loader, device, threshold=0.5, max_batches=None,
             desc='eval'):
    model.eval()
    metrics = SegMetrics(threshold=threshold)
    for k, (x, y, label, method) in enumerate(tqdm(loader, desc=desc,
                                                   leave=False)):
        if max_batches and k >= max_batches:
            break
        with torch.autocast(device.type, dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            logits = model(x.to(device, non_blocking=True))
        metrics.update(torch.sigmoid(logits.float()), y, label, method)
    return metrics.summary()


def eval_loader(cfg, split, fraction=1.0, degrade=None):
    ds = FaceSegDataset(os.path.join(cfg['index_dir'], split + '.csv'),
                        cfg['data_root'],
                        eval_transform(cfg['image_size'], degrade=degrade),
                        fraction=fraction)
    return DataLoader(ds, batch_size=cfg['batch_size'] * 2, shuffle=False,
                      num_workers=cfg['num_workers'], pin_memory=True,
                      persistent_workers=cfg['num_workers'] > 0)


def load_checkpoint(path, compression='c23', device=None):
    """Mô hình đã nạp trọng số, cfg (đổi thư mục dữ liệu nếu c40) và checkpoint."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    if compression != 'c23':
        cfg = dict(cfg, data_root='data/faces_' + compression,
                   index_dir='data/index_' + compression)
    device = device or torch.device('cuda' if torch.cuda.is_available()
                                    else 'cpu')
    model = build_model(cfg['arch'], cfg['encoder'], pretrained=False)
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval(), cfg, ckpt, device


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('checkpoint')
    parser.add_argument('--split', default='test',
                        choices=['val', 'test', 'dfd'])
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='Chỉ đánh giá một phần video (để chạy thử).')
    parser.add_argument('--compression', default='c23', choices=['c23', 'c40'],
                        help='c40: test độ bền trên data/faces_c40, data/index_c40.')
    args = parser.parse_args()

    model, cfg, ckpt, device = load_checkpoint(args.checkpoint, args.compression)
    tag = args.split
    if args.compression != 'c23':
        tag = '{}_{}'.format(args.split, args.compression)

    summary = run_eval(model, eval_loader(cfg, args.split, args.fraction),
                       device, cfg['threshold'], desc=tag)
    print('{} (epoch {}) on {}:'.format(cfg['name'], ckpt['epoch'], tag))
    print(format_table(summary))
    out = os.path.join(os.path.dirname(args.checkpoint),
                       'eval_{}.json'.format(tag))
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'epoch': ckpt['epoch'], 'split': args.split,
                   'compression': args.compression,
                   'fraction': args.fraction, 'metrics': summary}, f, indent=2)
    print('saved', out)


if __name__ == '__main__':
    main()
