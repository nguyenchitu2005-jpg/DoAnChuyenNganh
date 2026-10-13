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
           'DFD', 'actors', 'SBI']
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


def eval_transform(size=256, normalize=True, degrade=None):
    """`degrade`: làm hỏng ảnh test để đo độ bền, dạng 'jpeg:<chất lượng>',
    'blur:<kích thước kernel lẻ>' hoặc 'noise:<độ lệch chuẩn, 0-1>'. Chỉ áp cho ảnh."""
    return A.Compose([A.Resize(size, size)] + _degrade(degrade)
                     + _to_tensor(normalize))


def _degrade(spec):
    if not spec:
        return []
    kind, value = spec.split(':')
    if kind == 'jpeg':
        q = int(value)
        return [A.ImageCompression(quality_range=(q, q), p=1)]
    if kind == 'blur':
        k = int(value)
        # sigma cố định theo kernel (công thức mặc định của OpenCV), để mức blur
        # tăng đều theo k thay vì ngẫu nhiên
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        return [A.GaussianBlur(blur_limit=(k, k), sigma_limit=(sigma, sigma), p=1)]
    if kind == 'noise':
        s = float(value)
        return [A.GaussNoise(std_range=(s, s), p=1)]
    raise ValueError('unknown degradation: ' + spec)


def self_blend(image):
    """Self-Blended Image (Shiohara & Yamasaki, CVPR 2022), bản không cần landmark.

    Ảnh mặt thật được biến đổi nhẹ (màu, độ nét, co giãn, dịch) rồi dán lại lên chính
    nó theo một elip ngẫu nhiên có méo biên và biên mờ: ảnh giả có mask pixel chính xác,
    mô hình phải học dấu vết đường ghép chung của deepfake thay vì riêng 4 phương pháp.
    Mặt nằm giữa ảnh cắt (khung mặt nới 1,3 lần) nên elip đặt quanh tâm ảnh.
    Trả về ảnh uint8 và mask 0/1.
    """
    h, w = image.shape[:2]
    r = np.random.uniform
    core = np.zeros((h, w), np.float32)
    cv2.ellipse(core, (int(w * r(0.45, 0.55)), int(h * r(0.45, 0.6))),
                (int(w * r(0.2, 0.32)), int(h * r(0.26, 0.38))), r(-15, 15), 0, 360,
                1, -1)
    # méo biên: trường dịch chuyển mượt, biên độ 5-20 px
    grid = np.mgrid[0:h, 0:w].astype(np.float32)
    shift = []
    for _ in range(2):
        f = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32), (0, 0), 20)
        shift.append(f / (np.abs(f).max() + 1e-6) * r(5, 20))
    core = cv2.remap(core, grid[1] + shift[0], grid[0] + shift[1], cv2.INTER_LINEAR)
    k = 2 * np.random.randint(7, 23) + 1
    soft = cv2.GaussianBlur(core, (k, k), 0)
    alpha = soft * np.random.choice([0.25, 0.5, 0.75, 1, 1, 1])

    src = image.astype(np.float32) * r(0.9, 1.1) + r(-20, 20, 3)
    if np.random.rand() < 0.5:
        src = cv2.GaussianBlur(src, (0, 0), r(0.5, 1.5))
    else:   # làm nét (unsharp mask)
        src = cv2.addWeighted(src, 1.6, cv2.GaussianBlur(src, (0, 0), 2), -0.6, 0)
    s = r(0.95, 1.05)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
    m[:, 2] += r(-0.03, 0.03, 2) * (w, h)
    src = cv2.warpAffine(src, m, (w, h), borderMode=cv2.BORDER_REFLECT)

    out = src * alpha[..., None] + image * (1 - alpha[..., None])
    return np.clip(out, 0, 255).astype(np.uint8), (soft > 0.5).astype(np.uint8)


def _to_tensor(normalize):
    return ([A.Normalize(MEAN, STD)] if normalize else []) + [ToTensorV2()]


class FaceSegDataset(Dataset):
    """Mỗi mẫu: ảnh (3,H,W), mask (1,H,W) 0/1, nhãn thật/giả, chỉ số phương pháp.

    Frame thật không có file mask: dùng mask đen toàn bộ.
    `fraction` < 1 lấy ngẫu nhiên một phần video (giữ trọn các frame của video).
    `sbi_prob` > 0 (chỉ dùng khi train): mỗi lần đọc một frame thật, với xác suất này
    biến nó thành ảnh giả tự ghép (`self_blend`), nhãn giả, phương pháp 'SBI'.
    """

    def __init__(self, csv_path, root, transform, fraction=1.0, seed=0, sbi_prob=0.0):
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
        self.sbi_prob = sbi_prob

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
        elif np.random.rand() < self.sbi_prob:
            image, mask = self_blend(image)
            label, method = 1, METHODS.index('SBI')
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
