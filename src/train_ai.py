"""Mô hình 2: phân loại frame mặt thật / do AI tạo (EfficientNet-B0), rồi gộp theo video.

Chạy (sau preprocess_ai.py): python src/train_ai.py
Thử nhanh: python src/train_ai.py --epochs 1 --fraction 0.1
Mỗi epoch đánh giá AUC theo frame trên val, lưu best.pt theo AUC theo video. Xong thì:
điểm video = trung bình xác suất AI các frame; ngưỡng chọn trên val theo độ chính xác cân
bằng; đo trên test (bộ sinh đã gặp, danh tính mới) và extra (Hallo3, MAGI-1: bộ sinh chưa
gặp). Kết quả: runs/ai_effb0/{best.pt, history.jsonl, video_rule.json}.
Grad-CAM (`gradcam`): vùng mô hình dựa vào để quyết định, không phải vùng bị làm giả.
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import balanced_sampler, eval_transform, train_transform

RUN_DIR = 'runs/ai_effb0'


class FrameDataset(Dataset):
    """Ảnh mặt 256×256 và nhãn (1 = do AI tạo); `fraction` < 1 lấy một phần video."""

    def __init__(self, split, transform, fraction=1.0):
        with open(os.path.join('data/index_ai', split + '.csv'), newline='',
                  encoding='utf-8') as f:
            self.rows = [(r['frame'], r['source'], r['video'], int(r['label']))
                         for r in csv.DictReader(f)]
        if fraction < 1:
            videos = sorted({r[2] for r in self.rows})
            keep = set(random.Random(0).sample(videos, round(len(videos) * fraction)))
            self.rows = [r for r in self.rows if r[2] in keep]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def labels(self):
        return [r[3] for r in self.rows]

    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(os.path.join('data', self.rows[i][0])),
                             cv2.COLOR_BGR2RGB)
        return self.transform(image=image)['image'], self.rows[i][3]


def build_model(pretrained=True):
    return timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=1)


@torch.no_grad()
def predict(model, ds, device, batch_size=64, workers=4):
    model.eval()
    probs = []
    for x, _ in tqdm(DataLoader(ds, batch_size=batch_size, num_workers=workers),
                     desc='eval', leave=False):
        with torch.autocast(device.type, dtype=torch.float16):
            probs.append(torch.sigmoid(model(x.to(device)).float())[:, 0].cpu())
    return torch.cat(probs).numpy()


def video_scores(ds, probs):
    """Điểm video = trung bình xác suất AI các frame; trả về {video: (điểm, nhãn, nguồn)}."""
    d = defaultdict(list)
    for r, p in zip(ds.rows, probs):
        d[(r[2], r[3], r[1])].append(p)
    return {k[0]: (float(np.mean(v)), k[1], k[2]) for k, v in d.items()}


def rule_stats(scores, t):
    s = np.array([v[0] for v in scores.values()])
    y = np.array([v[1] for v in scores.values()])
    out = {'videos_real': int((y == 0).sum()), 'videos_ai': int((y == 1).sum())}
    if (y == 0).any():
        out['real_correct'] = float((s[y == 0] < t).mean())
    if (y == 1).any():
        out['ai_correct'] = float((s[y == 1] >= t).mean())
    if 'real_correct' in out and 'ai_correct' in out:
        out['balanced_acc'] = (out['real_correct'] + out['ai_correct']) / 2
        out['auc'] = float(roc_auc_score(y, s))
    by_source = defaultdict(list)
    for v in scores.values():
        by_source[v[2]].append(v[0] >= t if v[1] else v[0] < t)
    out['correct_by_source'] = {k: float(np.mean(v)) for k, v in sorted(by_source.items())}
    return out


def gradcam(model, x):
    """Bản đồ Grad-CAM (H×W, 0-1) và xác suất AI cho một ảnh đã chuẩn hoá (1,3,H,W)."""
    model.eval()
    feats = model.forward_features(x)
    feats.retain_grad()
    logit = model.forward_head(feats)
    model.zero_grad()
    logit.sum().backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
    cam = cam[0, 0].detach()
    cam = (cam / cam.max()).cpu().numpy() if cam.max() > 0 else cam.cpu().numpy()
    return cam, torch.sigmoid(logit).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    os.makedirs(RUN_DIR, exist_ok=True)

    train_ds = FrameDataset('train', train_transform(256), args.fraction)
    loaders = {s: FrameDataset(s, eval_transform(256), args.fraction)
               for s in ('val', 'test', 'extra')}
    sampler = balanced_sampler(train_ds, seed=42)   # video thật ít hơn video AI
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=True)
    print('train {} frames, val {}, test {}, extra {}'.format(
        len(train_ds), *(len(d) for d in loaders.values())))

    device = torch.device('cuda')
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(loader), pct_start=0.05)
    scaler = torch.amp.GradScaler()
    best = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = None
        bar = tqdm(loader, desc='epoch {}/{}'.format(epoch, args.epochs))
        for x, y in bar:
            x, y = x.to(device, non_blocking=True), y.float().to(device)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.binary_cross_entropy_with_logits(model(x)[:, 0].float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
            bar.set_postfix(loss='{:.4f}'.format(running), refresh=False)
        val = loaders['val']
        probs = predict(model, val, device, workers=args.workers)
        frame_auc = roc_auc_score(val.labels(), probs)
        vs = video_scores(val, probs)
        video_auc = roc_auc_score([v[1] for v in vs.values()], [v[0] for v in vs.values()])
        print('epoch {}: loss {:.4f}, val AUC frame {:.4f}, video {:.4f}'.format(
            epoch, running, frame_auc, video_auc))
        with open(os.path.join(RUN_DIR, 'history.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps({'epoch': epoch, 'loss': running, 'val_frame_auc': frame_auc,
                                'val_video_auc': video_auc}) + '\n')
        if video_auc > best:
            best = video_auc
            torch.save({'model': model.state_dict(), 'epoch': epoch}, os.path.join(
                RUN_DIR, 'best.pt'))

    model.load_state_dict(torch.load(os.path.join(RUN_DIR, 'best.pt'))['model'])
    scores = {s: video_scores(d, predict(model, d, device, workers=args.workers))
              for s, d in loaders.items()}
    grid = np.unique([v[0] for v in scores['val'].values()])
    t = max(grid, key=lambda g: rule_stats(scores['val'], g)['balanced_acc'])
    rule = {'threshold': float(t), 'epoch': torch.load(os.path.join(
        RUN_DIR, 'best.pt'))['epoch']}
    for s in scores:
        rule[s] = rule_stats(scores[s], t)
    with open(os.path.join(RUN_DIR, 'video_rule.json'), 'w', encoding='utf-8') as f:
        json.dump(rule, f, indent=2)
    print(json.dumps(rule, indent=2))


if __name__ == '__main__':
    main()
