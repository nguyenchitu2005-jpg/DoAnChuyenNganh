"""Train một mô hình segmentation vùng giả mạo (U-Net, DeepLabV3+ hoặc SegFormer).

Chạy:  python src/train.py --config configs/unet.yaml
Thử nhanh: python src/train.py --config configs/unet.yaml --fraction 0.05 --epochs 1
Mỗi epoch: đánh giá trên val, lưu last.pt, và best.pt khi điểm
(IoU ảnh giả + tỉ lệ ảnh thật không bị báo nhầm) / 2 tốt hơn.
Log TensorBoard: tensorboard --logdir runs
"""
import argparse
import json
import os
import random
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import FaceSegDataset, balanced_sampler, train_transform
from evaluate import eval_loader, run_eval
from metrics import format_table
from models import build_model


def load_config(args):
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    for key in ('epochs', 'batch_size', 'lr', 'name', 'data_root', 'index_dir',
                'num_workers'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    cfg['fraction'] = args.fraction
    if args.fraction < 1 and args.name is None:
        cfg['name'] += '_frac{:g}'.format(args.fraction)
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', required=True)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', dest='batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--name')
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='Chỉ dùng một phần video train/val (để chạy thử).')
    # Đổi thư mục khi train ở máy khác (Kaggle: dữ liệu ở /kaggle/input)
    parser.add_argument('--data-root', dest='data_root')
    parser.add_argument('--index-dir', dest='index_dir')
    parser.add_argument('--num-workers', dest='num_workers', type=int)
    parser.add_argument('--runs-dir', default='runs')
    args = parser.parse_args()
    cfg = load_config(args)
    seed_everything(cfg['seed'])

    run_dir = os.path.join(args.runs_dir, cfg['name'])
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    train_ds = FaceSegDataset(os.path.join(cfg['index_dir'], 'train.csv'),
                              cfg['data_root'],
                              train_transform(cfg['image_size']),
                              fraction=cfg['fraction'], seed=cfg['seed'])
    sampler = balanced_sampler(train_ds, cfg.get('samples_per_epoch'),
                               seed=cfg['seed'])
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              sampler=sampler, num_workers=cfg['num_workers'],
                              pin_memory=True, drop_last=True,
                              persistent_workers=cfg['num_workers'] > 0)
    val_loader = eval_loader(cfg, 'val', cfg['fraction'])
    print('train {} frames ({} samples/epoch), val {} frames'.format(
        len(train_ds), len(sampler), len(val_loader.dataset)))

    device = torch.device('cuda')
    model = build_model(cfg['arch'], cfg['encoder']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                                  weight_decay=cfg['weight_decay'])
    steps = cfg['epochs'] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg['lr'], total_steps=steps, pct_start=0.05)
    # Dice bù việc pixel giả ít hơn nhiều pixel thật; BCE giữ ổn định trên mặt thật
    bce = torch.nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss('binary', from_logits=True)
    scaler = torch.amp.GradScaler()
    writer = SummaryWriter(run_dir)

    best_score, step = -1.0, 0
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        t0, running = time.time(), 0.0
        bar = tqdm(train_loader, desc='epoch {}/{}'.format(epoch, cfg['epochs']))
        for x, y, _, _ in bar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(x)
            logits = logits.float()
            loss = (cfg['bce_weight'] * bce(logits, y)
                    + cfg['dice_weight'] * dice(logits, y))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            running = 0.98 * running + 0.02 * loss.item() if step > 1 \
                else loss.item()
            if step % 50 == 0:
                writer.add_scalar('train/loss', running, step)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], step)
                bar.set_postfix(loss='{:.4f}'.format(running))

        summary = run_eval(model, val_loader, device, cfg['threshold'],
                           desc='val')
        fake, real = summary.get('fake_all', {}), summary.get('real_all', {})
        for k in ('iou', 'dice', 'f1', 'auc'):
            if k in fake:
                writer.add_scalar('val/' + k, fake[k], epoch)
        if 'false_alarm' in real:
            writer.add_scalar('val/false_alarm', real['false_alarm'], epoch)
        print('epoch {} done in {:.0f} min, loss {:.4f}'.format(
            epoch, (time.time() - t0) / 60, running))
        print(format_table(summary))

        # Chỉ IoU thì mô hình "tô cả khuôn mặt" cũng điểm cao (mask Face2Face,
        # NeuralTextures phủ gần cả mặt), nên chọn theo trung bình IoU ảnh giả
        # và tỉ lệ ảnh thật không bị báo nhầm.
        score = (fake.get('iou', 0) + 1 - real.get('false_alarm', 1)) / 2
        writer.add_scalar('val/score', score, epoch)
        ckpt = {'model': model.state_dict(), 'cfg': cfg, 'epoch': epoch,
                'val': summary, 'score': score}
        torch.save(ckpt, os.path.join(run_dir, 'last.pt'))
        if score > best_score:
            best_score = score
            torch.save(ckpt, os.path.join(run_dir, 'best.pt'))
            print('new best val score {:.4f}'.format(best_score))
        with open(os.path.join(run_dir, 'history.jsonl'), 'a',
                  encoding='utf-8') as f:
            f.write(json.dumps({'epoch': epoch, 'loss': running,
                                'score': score, 'val': summary}) + '\n')
    writer.close()


if __name__ == '__main__':
    main()
