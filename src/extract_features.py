"""Tính sẵn đặc trưng từng khung của encoder Swin-T (đóng băng) cho mô hình thời gian (temporal.py).

Chạy: python src/extract_features.py runs/b1_mix/best.pt
      (mặc định đọc data/clips + data/index_clips, cả c23 và c40, mọi split)
Mỗi khung lưu (float16, thứ tự đúng thứ tự dòng của <index>/<split>.csv):
  gap   : (3, 768) trung bình tầng 1/32 ở 3 mức zoom, cùng một Swin:
          mức 1 = ảnh mặt + nền (nguyên ảnh 224, đúng đầu vào đầu phân loại của B1),
          mức 2 = mặt sát (phần giữa 172 px ≈ khung mặt 1,0×) phóng lên 224,
          mức 3 = vùng miệng (khung MediaPipe) phóng lên 224
          (--mouth-root: thêm mức 4 = vùng miệng cắt từ khung hình gốc, mouth_crops.py → gap (4, 768))
  roi   : RoI align tầng 1/8 (192) và 1/16 (384) của mức 1 tại mắt trái, mắt phải, mũi, miệng (4 × 576)
  prob  : xác suất giả của đầu nhị phân B1 trên mức 1 (dòng so sánh "B1 trung bình khung")
Khung vùng lấy từ <index>/landmarks.csv (face_landmarks.py); khung MediaPipe không thấy mặt
dùng khung trung vị của cả tập. Kết quả: <run>/features/<split>[_c40].npz
"""
import argparse
import csv
import os

import cv2
import numpy as np
import torch
from torchvision.ops import roi_align
from tqdm import tqdm

from dataset import MEAN, STD
from evaluate import eval_loader, load_checkpoint

REGIONS = ['leye', 'reye', 'nose', 'mouth']


def region_boxes(path):
    """{frame: mảng 4×4 (x1,y1,x2,y2) theo REGIONS}. Khung MediaPipe không thấy mặt dùng trung
    vị các khung cùng video (mặt DFD hay lệch khỏi giữa ảnh), video không có khung nào thì
    dùng trung vị cả tập."""
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    boxes, per_video = {}, {}
    for r in rows:
        if r['yaw']:
            b = np.array([[float(r[f'{k}_{c}']) for c in ('x1', 'y1', 'x2', 'y2')]
                          for k in REGIONS], np.float32)
            boxes[r['frame']] = b
            per_video.setdefault(os.path.dirname(r['frame']), []).append(b)
    median = np.median(np.stack(list(boxes.values())), 0)
    video_median = {v: np.median(np.stack(bs), 0) for v, bs in per_video.items()}
    return {r['frame']: boxes.get(r['frame'], video_median.get(os.path.dirname(r['frame']), median))
            for r in rows}, median


def load_mouths(root, names, device):
    """Ảnh vùng miệng độ phân giải cao (mouth_crops.py), chuẩn hoá như dataset."""
    m = np.stack([cv2.cvtColor(cv2.imread(os.path.join(root, f)), cv2.COLOR_BGR2RGB) for f in names])
    m = (m.astype(np.float32) / 255 - np.float32(MEAN)) / np.float32(STD)
    return torch.from_numpy(m).permute(0, 3, 1, 2).to(device)


@torch.no_grad()
def extract(model, loader, boxes_of, frames, device, mouth_root=None):
    gaps, rois, probs, k = [], [], [], 0
    for x, *_ in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        names = frames[k:k + len(x)]
        b = torch.from_numpy(np.stack([boxes_of[f] for f in names])).to(device)
        k += len(x)
        n, size = len(x), x.shape[-1]
        c = (size - size / 1.3) / 2                           # mức 2: bỏ phần nới 1,3× quanh mặt
        center = torch.tensor([c, c, size - c, size - c], device=device).expand(n, 4)
        per_image = torch.stack([center, b[:, REGIONS.index('mouth')]], 1)   # (N, 2, 4)
        # roi_align trên chính ảnh = cắt khung rồi phóng về size×size (nội suy song tuyến);
        # kết quả theo thứ tự ảnh 0 (mức 2, mức 3), ảnh 1 (mức 2, mức 3), ...
        crops = roi_align(x, list(per_image), output_size=size, spatial_scale=1.0,
                          sampling_ratio=2, aligned=True)
        crops = crops.view(n, 2, *x.shape[1:]).transpose(0, 1).flatten(0, 1)   # [mức 2..., mức 3...]
        if mouth_root:
            crops = torch.cat([crops, load_mouths(mouth_root, names, device)])   # [..., mức 4...]
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            feats = model.net.backbone.forward_with_filtered_kwargs(x).feature_maps
            pooled = feats[-1].mean((2, 3))
            prob = torch.sigmoid(model.bin_head(pooled)[:, 0].float())
            zpool = model.net.backbone.forward_with_filtered_kwargs(crops).feature_maps[-1].mean((2, 3))
        pooled = torch.stack([pooled, *zpool.split(n)], 1)      # (N, 3 hoặc 4, 768)
        rb = [t for t in b]                                   # danh sách N mảng 4×4
        parts = [roi_align(feats[i].float(), rb, output_size=2, spatial_scale=s, aligned=True)
                 .mean((2, 3)).view(len(x), len(REGIONS), -1) for i, s in ((1, 1 / 8), (2, 1 / 16))]
        gaps.append(pooled.float().cpu().half())
        rois.append(torch.cat(parts, 2).flatten(1).cpu().half())
        probs.append(prob.cpu())
    assert k == len(frames)
    return torch.cat(gaps).numpy(), torch.cat(rois).numpy(), torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--root', default='data/clips')
    ap.add_argument('--index', default='data/index_clips')
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test', 'dfd'])
    ap.add_argument('--compression', nargs='+', default=['c23', 'c40'])
    ap.add_argument('--fraction', type=float, default=1.0)
    ap.add_argument('--out-dir', dest='out_dir', help='mặc định <run>/features')
    ap.add_argument('--mouth-root', dest='mouth_root',
                    help='ảnh vùng miệng cắt từ khung hình gốc (mouth_crops.py --out), bản c40 '
                         'lấy <mouth-root>_c40; thêm mức zoom 4')
    args = ap.parse_args()
    boxes_of, _ = region_boxes(os.path.join(args.index, 'landmarks.csv'))
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint), 'features')
    os.makedirs(out_dir, exist_ok=True)
    for comp in args.compression:
        model, cfg, _, device = load_checkpoint(args.checkpoint)
        assert getattr(model, 'multitask', False), 'cần mô hình có đầu phân loại (B1)'
        suffix = '' if comp == 'c23' else '_' + comp
        cfg = dict(cfg, data_root=args.root + suffix, index_dir=args.index + suffix, mix_c40=False)
        for split in args.splits:
            out = os.path.join(out_dir, f'{split}{suffix}.npz')
            if os.path.isfile(out):
                continue
            loader = eval_loader(cfg, split, args.fraction)
            frames = [r[0] for r in loader.dataset.rows]          # loader không xáo trộn
            mouth = args.mouth_root and args.mouth_root + suffix
            gap, roi, prob = extract(model, loader, boxes_of, frames, device, mouth)
            np.savez(out, frames=np.array(frames), gap=gap, roi=roi, prob=prob)
            print(f'{split}{suffix}: {len(frames)} khung → {out}')


if __name__ == '__main__':
    main()
