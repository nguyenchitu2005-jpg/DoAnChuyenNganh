"""Đo độ bền: đánh giá một checkpoint trên ảnh test bị nén JPEG, blur, noise ở nhiều mức.

Chạy: python src/robustness.py runs/unet_effb0/best.pt
Thử nhanh: python src/robustness.py runs/unet_effb0/best.pt --fraction 0.05
Kết quả trong <thư mục run>/robustness/: robustness.csv, robustness.md (bảng),
robustness.png (IoU ảnh giả và tỉ lệ báo nhầm theo từng mức làm hỏng).
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from dataset import MEAN, STD, FaceSegDataset, _degrade, eval_transform  # noqa: E402
from evaluate import load_checkpoint  # noqa: E402
from metrics import SegMetrics  # noqa: E402
import albumentations as A  # noqa: E402

FAKE_METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
# Mức làm hỏng; mức đầu mỗi loại là ảnh gốc (không làm hỏng)
LEVELS = {
    'jpeg': [None, 90, 70, 50, 30, 10],
    'blur': [None, 3, 7, 11, 15, 21],   # ảnh mặt phóng từ ~130 px nên đã hơi mờ
    'noise': [None, 0.02, 0.05, 0.1, 0.15],
}
AXIS = {'jpeg': 'Chất lượng JPEG', 'blur': 'Kích thước kernel blur',
        'noise': 'Độ lệch chuẩn noise'}
CLEAN_X = {'jpeg': 100, 'blur': 1, 'noise': 0}  # vị trí của ảnh gốc trên trục x

# Bảng màu mặc định (dataviz, đã kiểm tra bằng validate_palette.js), thứ tự cố định
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
MARKERS = ['o', 's', '^', 'D']
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID, AXIS_LINE = '#e1e0d9', '#c3c2b7'


@torch.no_grad()
def evaluate_levels(model, cfg, device, split, fraction):
    """Đọc dữ liệu một lần; mỗi batch được làm hỏng ở mọi mức rồi đưa qua mô hình
    (tạo DataLoader riêng cho từng mức tốn khoảng 45 giây khởi động worker trên Windows)."""
    specs = [None] + ['{}:{}'.format(k, v) for k, levels in LEVELS.items()
                      for v in levels if v is not None]
    degraders = {s: A.Compose(_degrade(s)) for s in specs if s}
    metrics = {s: SegMetrics(threshold=cfg['threshold']) for s in specs}
    ds = FaceSegDataset(os.path.join(cfg['index_dir'], split + '.csv'),
                        cfg['data_root'],
                        eval_transform(cfg['image_size'], normalize=False),
                        fraction=fraction)
    loader = DataLoader(ds, batch_size=cfg['batch_size'] * 2, shuffle=False,
                        num_workers=cfg['num_workers'], pin_memory=True)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1) * 255
    for x, y, label, method in tqdm(loader, desc='robustness'):
        images = np.ascontiguousarray(x.permute(0, 2, 3, 1).numpy())  # uint8 RGB
        for s in specs:
            batch = images if s is None else np.stack(
                [degraders[s](image=im)['image'] for im in images])
            t = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).float()
            with torch.autocast(device.type, dtype=torch.float16,
                                enabled=device.type == 'cuda'):
                logits = model((t - mean) / std)
            metrics[s].update(torch.sigmoid(logits.float()), y, label, method)
    summaries = {s: m.summary() for s, m in metrics.items()}

    rows = []
    for kind, levels in LEVELS.items():
        for level in levels:
            spec = None if level is None else '{}:{}'.format(kind, level)
            summary = summaries[spec]
            row = {'kind': kind,
                   'level': 'gốc' if level is None else level,
                   'x': CLEAN_X[kind] if level is None else level}
            for m in FAKE_METHODS:
                row['iou_' + m] = summary.get(m, {}).get('iou')
            fake, real = summary.get('fake_all', {}), summary.get('real_all', {})
            row.update(iou=fake.get('iou'), dice=fake.get('dice'),
                       f1=fake.get('f1'), auc=fake.get('auc'),
                       false_alarm=real.get('false_alarm'))
            rows.append(row)
            print('{:<6} {:>5}  IoU {}  FA {}'.format(
                kind, str(row['level']),
                '-' if row['iou'] is None else '{:.3f}'.format(row['iou']),
                '-' if row['false_alarm'] is None
                else '{:.1%}'.format(row['false_alarm'])))
    return rows


def write_tables(rows, out_dir):
    keys = list(rows[0].keys())
    with open(os.path.join(out_dir, 'robustness.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    head = (['Làm hỏng', 'Mức'] + ['IoU ' + m for m in FAKE_METHODS]
            + ['IoU tất cả', 'Dice', 'F1', 'AUC', 'Báo nhầm'])
    lines = ['| ' + ' | '.join(head) + ' |', '|' + ' --- |' * len(head)]
    def fmt(v, spec='{:.3f}'):
        return '-' if v is None else spec.format(v)

    for r in rows:
        cells = [r['kind'], str(r['level'])]
        cells += [fmt(r['iou_' + m]) for m in FAKE_METHODS]
        cells += [fmt(r[k]) for k in ('iou', 'dice', 'f1', 'auc')]
        cells.append(fmt(r['false_alarm'], '{:.1%}'))
        lines.append('| ' + ' | '.join(cells) + ' |')
    with open(os.path.join(out_dir, 'robustness.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS_LINE)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot(rows, out_path, title):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey='row', sharex='col',
                             gridspec_kw={'height_ratios': [3, 2]})
    fig.patch.set_facecolor(SURFACE)
    for col, kind in enumerate(LEVELS):
        rs = [r for r in rows if r['kind'] == kind]
        xs = [r['x'] for r in rs]
        ax = axes[0, col]
        style(ax)
        ends = []
        for i, m in enumerate(FAKE_METHODS):
            ys = [np.nan if r['iou_' + m] is None else r['iou_' + m] for r in rs]
            ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker=MARKERS[i],
                    markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=m)
            if not np.isnan(ys[-1]):
                ends.append([ys[-1], m])
        if col == len(LEVELS) - 1:
            # nhãn trực tiếp ở cuối đường, giãn ra để không chồng nhau
            ends.sort()
            for k in range(1, len(ends)):
                ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.06)
            for y, m in ends:
                ax.annotate(m, (xs[-1], y), xytext=(8, 0),
                            textcoords='offset points', va='center',
                            fontsize=9, color=INK2, annotation_clip=False)
        ax.set_ylim(0, 1)
        ax.set_title(AXIS[kind], color=INK, fontsize=11, loc='left')
        if col == 0:
            ax.set_ylabel('IoU ảnh giả', color=INK2)

        ax = axes[1, col]
        style(ax)
        ax.plot(xs, [np.nan if r['false_alarm'] is None
                     else 100 * r['false_alarm'] for r in rs], color=SERIES[0],
                linewidth=2, marker='o', markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(AXIS[kind] + ('  (mức 100 = ảnh gốc)' if kind == 'jpeg'
                                    else '  (mức đầu = ảnh gốc)'),
                      color=MUTED, fontsize=9)
        if col == 0:
            ax.set_ylabel('Báo nhầm mặt thật (%)', color=INK2)
        if kind == 'jpeg':
            ax.invert_xaxis()   # trái sang phải = càng hỏng (trục x dùng chung cột)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=4, frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(title, x=0.01, ha='left', color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.95, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('checkpoint')
    parser.add_argument('--split', default='test', choices=['val', 'test', 'dfd'])
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='Chỉ đánh giá một phần video (để chạy thử).')
    parser.add_argument('--cpu', action='store_true',
                        help='Chạy trên CPU (khi GPU đang bận train).')
    args = parser.parse_args()

    model, cfg, ckpt, device = load_checkpoint(
        args.checkpoint, device=torch.device('cpu') if args.cpu else None)
    if args.cpu:
        # worker DataLoader + mô hình trên CPU bị segfault trên Windows
        cfg = dict(cfg, num_workers=0)
    out_dir = os.path.join(os.path.dirname(args.checkpoint), 'robustness')
    os.makedirs(out_dir, exist_ok=True)
    rows = evaluate_levels(model, cfg, device, args.split, args.fraction)
    write_tables(rows, out_dir)
    plot(rows, os.path.join(out_dir, 'robustness.png'),
         '{}: độ bền trên FF++ {} (epoch {})'.format(cfg['name'], args.split,
                                                    ckpt['epoch']))
    print('saved', out_dir)


if __name__ == '__main__':
    main()
