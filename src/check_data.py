"""Kiểm tra dữ liệu: chồng mask lên ảnh sau augment, đếm tỉ lệ thật/giả mỗi batch.

Chạy: python src/check_data.py --n 200
Ảnh lưu ở runs/check_data/: mỗi ảnh là một lưới 8 mẫu (vùng đỏ = mask giả).
"""
import argparse
import os

import cv2
import numpy as np
from torch.utils.data import DataLoader

from dataset import METHODS, FaceSegDataset, balanced_sampler, train_transform


def overlay(image, mask):
    image = image.permute(1, 2, 0).numpy().astype(np.uint8)[:, :, ::-1].copy()
    m = mask[0].numpy() > 0.5
    image[m] = (0.5 * image[m] + [0, 0, 127]).astype(np.uint8)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', default='data/faces')
    parser.add_argument('--csv', default='data/index/train.csv')
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--out', default='runs/check_data')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds = FaceSegDataset(args.csv, args.root, train_transform(normalize=False))
    loader = DataLoader(ds, batch_size=8, sampler=balanced_sampler(ds),
                        num_workers=4)
    real, seen = 0, 0
    for k, (x, y, label, method) in enumerate(loader):
        tiles = [overlay(x[i], y[i]) for i in range(len(x))]
        for i, t in enumerate(tiles):
            cv2.putText(t, METHODS[int(method[i])], (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(args.out, 'batch_{:03d}.jpg'.format(k)),
                    np.hstack(tiles))
        real += int((label == 0).sum())
        seen += len(label)
        # ảnh giả phải có vùng mask, ảnh thật mask phải đen
        bad = [(METHODS[int(method[i])], float(y[i].mean())) for i in range(len(y))
               if (label[i] == 1) != bool(y[i].sum())]
        if bad:
            print('batch {}: mask lạ {}'.format(k, bad))
        if seen >= args.n:
            break
    print('{} mẫu, {:.0%} thật; ảnh lưu ở {}'.format(seen, real / seen, args.out))


if __name__ == '__main__':
    main()
