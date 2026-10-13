"""Vẽ hình so sánh mask dự đoán với mask thật, cho báo cáo và phân tích ca sai.

Chạy: python src/visualize.py runs/unet_effb0/best.pt --split test --mode random
      python src/visualize.py runs/unet_effb0/best.pt --split test --mode worst
      (--cpu khi GPU đang bận train; --compression c40 để xem trên c40)
Mỗi phương pháp giả một hình, mỗi hàng một ảnh: ảnh | mask thật | xác suất dự đoán |
sai số (đúng / tô thừa / bỏ sót). Ảnh thật: ảnh | xác suất | vùng bị báo nhầm.
--mode worst chọn ảnh có IoU thấp nhất (ảnh giả) và vùng báo nhầm lớn nhất (ảnh thật),
cần chạy mô hình trên toàn bộ tập trước để xếp hạng.
Kết quả: <thư mục run>/figures/<split>[_c40]_<mode>_<phương pháp>.png
"""
import argparse
import os
import random
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from dataset import METHODS, FaceSegDataset, eval_transform  # noqa: E402
from evaluate import load_checkpoint  # noqa: E402

# Bảng màu mặc định (dataviz): xanh dương cho mask, 3 màu đầu cho đúng/thừa/sót
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
# Thang xác suất một màu, nhạt đến đậm (dải xanh dương của bảng màu)
PROB_CMAP = LinearSegmentedColormap.from_list(
    'prob', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#0d366b'])


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def tint(image, mask, color, alpha=0.5):
    out = image.astype(float).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * hex_rgb(color)
    return out.astype(np.uint8)


def error_map(image, pred, gt):
    """Ảnh xám, tô xanh dương = đúng, cam = tô thừa, xanh ngọc = bỏ sót."""
    gray = np.repeat(image.mean(axis=2, keepdims=True), 3, axis=2)
    out = tint(gray, pred & gt, BLUE, 0.65)
    out = tint(out, pred & ~gt, ORANGE, 0.75)
    return tint(out, ~pred & gt, AQUA, 0.75)


@torch.no_grad()
def predict(model, x, device):
    with torch.autocast(device.type, dtype=torch.float16,
                        enabled=device.type == 'cuda'):
        return torch.sigmoid(model(x.to(device)).float()).cpu()


@torch.no_grad()
def rank_all(model, ds, cfg, device, threshold):
    """Điểm mỗi ảnh: ảnh giả = IoU, ảnh thật = phần trăm diện tích bị báo nhầm;
    kèm diện tích dự đoán (cả ảnh giả) để analyze.py kết luận theo video."""
    loader = DataLoader(ds, batch_size=cfg['batch_size'] * 2, shuffle=False,
                        num_workers=cfg['num_workers'])
    scores, start = [], 0
    for x, y, label, method in tqdm(loader, desc='rank'):
        pred = predict(model, x, device) > threshold
        gt = y > 0.5
        inter = (pred & gt).flatten(1).sum(1).float()
        union = (pred | gt).flatten(1).sum(1).float()
        area = pred.flatten(1).float().mean(1)
        for i in range(len(x)):
            s = (inter[i] / union[i]).item() if label[i] else area[i].item()
            scores.append((start + i, METHODS[int(method[i])], int(label[i]), s,
                           area[i].item()))
        start += len(x)
    return scores


def pick(ds, scores, mode, n, seed=0):
    by_method = defaultdict(list)
    if scores is None:
        for i, r in enumerate(ds.rows):
            by_method[METHODS[r[2]]].append((i, r[4], None))
        rng = random.Random(seed)
        return {m: rng.sample(v, min(n, len(v))) for m, v in by_method.items()}
    for i, m, label, s, _ in scores:
        by_method[m].append((i, label, s))
    out = {}
    for m, v in by_method.items():
        # ảnh giả: IoU thấp nhất trước; ảnh thật: diện tích báo nhầm lớn nhất trước;
        # mỗi video một frame để thấy nhiều kiểu lỗi khác nhau
        seen, chosen = set(), []
        for item in sorted(v, key=lambda t: t[2] if t[1] else -t[2]):
            video = ds.rows[item[0]][3]
            if video not in seen:
                seen.add(video)
                chosen.append(item)
            if len(chosen) == n:
                break
        out[m] = chosen
    return out


def render(model, ds_model, ds_view, items, method, device, threshold, path,
           title):
    fake = bool(items[0][1])
    cols = (['Ảnh', 'Mask thật', 'Xác suất dự đoán', 'Sai số'] if fake
            else ['Ảnh', 'Xác suất dự đoán', 'Vùng bị báo nhầm'])
    fig, axes = plt.subplots(len(items), len(cols),
                             figsize=(2.6 * len(cols), 2.6 * len(items) + 0.9),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for row, (i, label, _) in enumerate(items):
        x, y, _, _ = ds_model[i]
        image = ds_view[i][0].permute(1, 2, 0).numpy()
        prob = predict(model, x[None], device)[0, 0].numpy()
        pred, gt = prob > threshold, y[0].numpy() > 0.5
        if fake:
            union = (pred | gt).sum()
            score = 'IoU {:.2f}'.format((pred & gt).sum() / union if union else 1.0)
            panels = [image, tint(image, gt, BLUE), prob, error_map(image, pred, gt)]
        else:
            score = 'báo nhầm {:.0%} diện tích'.format(pred.mean())
            panels = [image, prob, tint(image, pred, ORANGE, 0.6)]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            if panel is prob:
                ax.imshow(prob, cmap=PROB_CMAP, vmin=0, vmax=1)
            else:
                ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                # viền mảnh cho bản đồ xác suất, để ô toàn 0 không lẫn vào nền
                s.set_visible(panel is prob)
                s.set_color('#c3c2b7')
            if row == 0:
                ax.set_title(cols[col], color=INK, fontsize=10)
        frame, video = ds_model.rows[i][0], ds_model.rows[i][3].split('/', 1)[1]
        axes[row, 0].set_ylabel('{}\nframe {}\n{}'.format(
            video[:24], os.path.splitext(os.path.basename(frame))[0], score),
            color=INK2, fontsize=8)
    if fake:
        handles = [Patch(color=BLUE, label='Đúng (TP)'),
                   Patch(color=ORANGE, label='Tô thừa (FP)'),
                   Patch(color=AQUA, label='Bỏ sót (FN)')]
        fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
                   fontsize=9, labelcolor=INK2)
    fig.suptitle(title, x=0.01, ha='left', color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.03 if fake else 0, 1, 0.97))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('checkpoint')
    parser.add_argument('--split', default='test', choices=['val', 'test', 'dfd'])
    parser.add_argument('--compression', default='c23', choices=['c23', 'c40'])
    parser.add_argument('--mode', default='random', choices=['random', 'worst'])
    parser.add_argument('--n', type=int, default=6, help='Số ảnh mỗi phương pháp.')
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--cpu', action='store_true',
                        help='Chạy trên CPU (khi GPU đang bận train).')
    args = parser.parse_args()

    model, cfg, ckpt, device = load_checkpoint(
        args.checkpoint, args.compression,
        torch.device('cpu') if args.cpu else None)
    if args.cpu:
        cfg = dict(cfg, num_workers=0)   # worker + mô hình trên CPU bị segfault
    csv_path = os.path.join(cfg['index_dir'], args.split + '.csv')
    ds_model = FaceSegDataset(csv_path, cfg['data_root'],
                              eval_transform(cfg['image_size']), args.fraction)
    ds_view = FaceSegDataset(csv_path, cfg['data_root'],
                             eval_transform(cfg['image_size'], normalize=False),
                             args.fraction)
    scores = (rank_all(model, ds_model, cfg, device, cfg['threshold'])
              if args.mode == 'worst' else None)

    out_dir = os.path.join(os.path.dirname(args.checkpoint), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    tag = args.split + ('' if args.compression == 'c23' else '_' + args.compression)
    what = {'random': 'ngẫu nhiên', 'worst': 'ca sai nhiều nhất'}[args.mode]
    for method, items in pick(ds_model, scores, args.mode, args.n).items():
        path = os.path.join(out_dir, '{}_{}_{}.png'.format(tag, args.mode, method))
        render(model, ds_model, ds_view, items, method, device, cfg['threshold'],
               path, '{} trên {} {}: {} ({})'.format(
                   cfg['name'], tag, method, what, 'epoch {}'.format(ckpt['epoch'])))
        print('saved', path)


if __name__ == '__main__':
    main()
