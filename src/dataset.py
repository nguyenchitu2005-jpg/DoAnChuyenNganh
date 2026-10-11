"""Dataset PyTorch cho segmentation vùng giả mạo trên ảnh mặt đã cắt (src/preprocess.py)."""
import csv
import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, WeightedRandomSampler

METHODS = ['original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures',
           'DFD', 'actors']
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MASK_THRESHOLD = 127  # mask đã nhị phân 0/255 lúc tiền xử lý


def train_transform(size=256, normalize=True):
    # Phép hình học áp dụng cho cả ảnh và mask (mask nội suy nearest);
    # nén, blur, noise, độ sáng chỉ áp dụng cho ảnh.
    steps = [
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-10, 10),
                 translate_percent=(-0.05, 0.05), p=0.5),
        A.OneOf([
            A.ImageCompression(quality_range=(40, 100)),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.GaussNoise(std_range=(0.02, 0.08)),
        ], p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Resize(size, size),
    ]
    return A.Compose(steps + _to_tensor(normalize))


def eval_transform(size=256, normalize=True):
    return A.Compose([A.Resize(size, size)] + _to_tensor(normalize))


def _to_tensor(normalize):
    return ([A.Normalize(MEAN, STD)] if normalize else []) + [ToTensorV2()]


class FaceSegDataset(Dataset):
    """Mỗi mẫu: ảnh (3,H,W), mask (1,H,W) 0/1, nhãn thật/giả, chỉ số phương pháp.

    Frame thật không có file mask: dùng mask đen toàn bộ.
    `fraction` < 1 lấy ngẫu nhiên một phần video (giữ trọn các frame của video).
    """

    def __init__(self, csv_path, root, transform, fraction=1.0, seed=0):
        with open(csv_path, newline='', encoding='utf-8') as f:
            rows = [(r['frame'], r['mask'], METHODS.index(r['method']),
                     r['method'] + '/' + r['video'], int(r['label']))
                    for r in csv.DictReader(f)]
        if fraction < 1:
            videos = sorted({r[3] for r in rows})
            keep = set(random.Random(seed).sample(
                videos, max(1, round(len(videos) * fraction))))
            rows = [r for r in rows if r[3] in keep]
        self.rows = rows
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def labels(self):
        return [r[4] for r in self.rows]

    def __getitem__(self, i):
        frame, mask_path, method, _, label = self.rows[i]
        image = cv2.imread(os.path.join(self.root, frame), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if label:
            mask = cv2.imread(os.path.join(self.root, mask_path),
                              cv2.IMREAD_GRAYSCALE)
            mask = (mask > MASK_THRESHOLD).astype(np.uint8)
        else:
            mask = np.zeros(image.shape[:2], np.uint8)
        out = self.transform(image=image, mask=mask)
        return out['image'], out['mask'][None].float(), label, method


def balanced_sampler(dataset, num_samples=None, seed=0):
    """Lấy mẫu có hoàn lại để mỗi batch có trung bình 50% frame thật, 50% giả.

    Frame thật (khoảng 20% dữ liệu) được lặp lại khoảng 4 lần mỗi epoch.
    """
    labels = torch.tensor(dataset.labels())
    n_fake = int(labels.sum())
    n_real = len(labels) - n_fake
    weights = torch.where(labels == 1, 0.5 / n_fake, 0.5 / n_real).double()
    return WeightedRandomSampler(weights, num_samples or len(labels),
                                 replacement=True,
                                 generator=torch.Generator().manual_seed(seed))
