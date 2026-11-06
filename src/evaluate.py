"""Đánh giá một checkpoint trên val, test (FF++) hoặc dfd (kiểm tra chéo).

Chạy: python src/evaluate.py runs/<tên>/best.pt --split test
Độ bền: python src/evaluate.py runs/<tên>/best.pt --split test --compression c40
Kết quả in ra bảng theo từng phương pháp và lưu vào <thư mục run>/eval_<split>[_c40][_cascade].json
Nối tầng (mô hình đa nhiệm): thêm --cascade, ảnh mà đầu nhị phân đoán là thật có mask rỗng.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FaceSegDataset, eval_transform, split_sources
from metrics import SegMetrics, cls_metrics, format_cls, format_table
from models import build_model


@torch.no_grad()
def run_eval(model, loader, device, threshold=0.5, max_batches=None,
             desc='eval', cascade=False):
    """Chỉ số segmentation theo phương pháp; mô hình đa nhiệm có thêm khoá 'cls'.
    `cascade`: phân loại trước, ảnh có xác suất giả < 0,5 thì mask rỗng."""
    model.eval()
    metrics = SegMetrics(threshold=threshold)
    multitask = getattr(model, 'multitask', False)
    cls_out = []
    for k, (x, y, label, method) in enumerate(tqdm(loader, desc=desc,
                                                   leave=False)):
        if max_batches and k >= max_batches:
            break
        with torch.autocast(device.type, dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            out = model(x.to(device, non_blocking=True), with_cls=multitask)
        logits = out[0] if multitask else out
        probs = torch.sigmoid(logits.float())
        if cascade:
            probs = probs * (torch.sigmoid(out[1].float()) >= 0.5)[:, None, None, None]
        metrics.update(probs, y, label, method)
        if multitask:
            cls_out.append((torch.sigmoid(out[1].float()).cpu(),
                            torch.softmax(out[2].float(), 1).cpu(), label, method))
    summary = metrics.summary()
    if multitask:
        bin_prob, cls_prob, labels, methods = (torch.cat(t).numpy()
                                               for t in zip(*cls_out))
        # loader không xáo trộn nên thứ tự ảnh trùng thứ tự dòng trong csv
        videos = [r[3] for r in loader.dataset.rows[:len(labels)]]
        summary['cls'] = cls_metrics(bin_prob, cls_prob, labels, methods, videos)
    return summary


def eval_loader(cfg, split, fraction=1.0, degrade=None):
    ds = FaceSegDataset(split_sources(cfg, split), None,
                        eval_transform(cfg['image_size'], degrade=degrade),
                        fraction=fraction)
    return DataLoader(ds, batch_size=cfg['batch_size'] * 2, shuffle=False,
                      num_workers=cfg['num_workers'], pin_memory=True,
                      persistent_workers=cfg['num_workers'] > 0)


def load_checkpoint(path, compression='c23', device=None):
    """Mô hình đã nạp trọng số, cfg (đổi thư mục dữ liệu nếu c40) và checkpoint."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = dict(ckpt['cfg'], mix_c40=False)   # đánh giá từng mức nén riêng, kể cả mô hình train trộn
    if not os.path.isdir(cfg['data_root']):
        # checkpoint train ở máy khác (Kaggle): dùng dữ liệu ở máy này
        cfg = dict(cfg, data_root='data/faces', index_dir='data/index',
                   num_workers=6)
    if compression != 'c23':
        cfg = dict(cfg, data_root='data/faces_' + compression,
                   index_dir='data/index_' + compression)
    device = device or torch.device('cuda' if torch.cuda.is_available()
                                    else 'cpu')
    model = build_model(cfg['arch'], cfg['encoder'], pretrained=False,
                        multitask=cfg.get('multitask', False),
                        cnn_branch=cfg.get('cnn_branch'))
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
    parser.add_argument('--cascade', action='store_true',
                        help='Phân loại trước rồi mới tô vùng (mô hình đa nhiệm).')
    args = parser.parse_args()

    model, cfg, ckpt, device = load_checkpoint(args.checkpoint, args.compression)
    tag = args.split
    if args.compression != 'c23':
        tag = '{}_{}'.format(args.split, args.compression)
    if args.cascade:
        if not model.multitask:
            raise SystemExit('--cascade cần mô hình có đầu phân loại (multitask: true).')
        tag += '_cascade'

    summary = run_eval(model, eval_loader(cfg, args.split, args.fraction),
                       device, cfg['threshold'], desc=tag, cascade=args.cascade)
    print('{} (epoch {}) on {}:'.format(cfg['name'], ckpt['epoch'], tag))
    print(format_table(summary))
    if 'cls' in summary:
        print(format_cls(summary['cls']))
    out = os.path.join(os.path.dirname(args.checkpoint),
                       'eval_{}.json'.format(tag))
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'epoch': ckpt['epoch'], 'split': args.split,
                   'compression': args.compression, 'cascade': args.cascade,
                   'fraction': args.fraction, 'metrics': summary}, f, indent=2)
    print('saved', out)


if __name__ == '__main__':
    main()
