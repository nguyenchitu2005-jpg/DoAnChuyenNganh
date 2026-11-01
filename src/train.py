"""Train một mô hình segmentation vùng giả mạo (kiến trúc trong models.py).

Chạy:  python src/train.py --config configs/<tên>.yaml
Thử nhanh: python src/train.py --config configs/<tên>.yaml --fraction 0.05 --epochs 1
Mỗi epoch: đánh giá trên val, lưu last.pt, và best.pt khi điểm
(IoU ảnh giả + tỉ lệ ảnh thật không bị báo nhầm) / 2 tốt hơn.
Dừng giữa chừng (Ctrl+C, tắt máy) rồi train tiếp từ epoch cuối đã xong:
    python src/train.py --config configs/<tên>.yaml --resume
Giới hạn thời gian mỗi phiên (Kaggle): thêm --max-hours 11.
Fine-tune từ mô hình khác: khoá `init: <best.pt>` trong config (chỉ nạp trọng số).
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
import torch.nn.functional as F
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
    parser.add_argument('--resume', action='store_true',
                        help='Train tiếp từ epoch cuối đã xong (last.pt của run cùng tên).')
    parser.add_argument('--max-hours', dest='max_hours', type=float,
                        help='Dừng sau epoch nếu epoch kế tiếp sẽ vượt mốc này '
                             '(Kaggle ngắt phiên ở 12 giờ); train tiếp bằng --resume.')
    args = parser.parse_args()
    t_start = time.time()
    cfg = load_config(args)
    seed_everything(cfg['seed'])

    run_dir = os.path.join(args.runs_dir, cfg['name'])
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    train_ds = FaceSegDataset(os.path.join(cfg['index_dir'], 'train.csv'),
                              cfg['data_root'],
                              train_transform(cfg['image_size']),
                              fraction=cfg['fraction'], seed=cfg['seed'],
                              sbi_prob=cfg.get('sbi_prob', 0.0))
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
    topk_weight = cfg.get('real_topk_weight', 0.0)
    scaler = torch.amp.GradScaler()
    writer = SummaryWriter(run_dir)

    best_score, step, start = -1.0, 0, 1
    last_path = os.path.join(run_dir, 'last.pt')
    if args.resume:
        last = torch.load(last_path, map_location=device, weights_only=False)
        if 'optimizer' not in last:
            raise SystemExit(last_path + ' lưu bằng bản train.py cũ, không có trạng thái '
                             'optimizer nên không train tiếp được.')
        model.load_state_dict(last['model'])
        optimizer.load_state_dict(last['optimizer'])
        scheduler.load_state_dict(last['scheduler'])
        scaler.load_state_dict(last['scaler'])
        best_score, step, start = last['best_score'], last['step'], last['epoch'] + 1
        print('train tiếp từ epoch {} (best val score {:.4f})'.format(start, best_score))
    elif cfg.get('init'):
        # fine-tune: chỉ lấy trọng số, optimizer và lịch lr bắt đầu mới
        model.load_state_dict(torch.load(cfg['init'], map_location=device,
                                         weights_only=False)['model'])
        print('khởi tạo từ', cfg['init'])
    for epoch in range(start, cfg['epochs'] + 1):
        model.train()
        t0, running = time.time(), 0.0
        bar = tqdm(train_loader, desc='epoch {}/{}'.format(epoch, cfg['epochs']))
        for x, y, label, _ in bar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(x)
            logits = logits.float()
            loss = (cfg['bce_weight'] * bce(logits, y)
                    + cfg['dice_weight'] * dice(logits, y))
            real = (label == 0).to(device)   # ảnh SBI có nhãn 1 nên không tính
            if topk_weight and real.any():
                # BCE trung bình mọi pixel gần như không phạt một mảng nhỏ tô nhầm trên mặt
                # thật, dù mảng > 1% diện tích đã là báo nhầm: phạt riêng 1% pixel cao nhất
                top = logits[real].flatten(1).topk(logits[0].numel() // 100, dim=1).values
                loss = loss + topk_weight * F.softplus(top).mean()   # = BCE với nhãn 0
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.get('grad_clip'):   # Swin dễ mất ổn định lúc đầu (KE_HOACH.md mục 12)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
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
        if score > best_score:
            best_score = score
            torch.save(ckpt, os.path.join(run_dir, 'best.pt'))
            print('new best val score {:.4f}'.format(best_score))
        # last.pt kèm trạng thái optimizer/scheduler để --resume; ghi file tạm rồi đổi
        # tên, nên dừng giữa lúc lưu cũng không làm hỏng last.pt cũ
        torch.save(dict(ckpt, optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                        best_score=best_score, step=step), last_path + '.tmp')
        os.replace(last_path + '.tmp', last_path)
        with open(os.path.join(run_dir, 'history.jsonl'), 'a',
                  encoding='utf-8') as f:
            f.write(json.dumps({'epoch': epoch, 'loss': running,
                                'score': score, 'val': summary}) + '\n')
        if args.max_hours and epoch < cfg['epochs'] and \
                time.time() + (time.time() - t0) - t_start > args.max_hours * 3600:
            print('dừng sau epoch {} cho kịp giờ; train tiếp bằng --resume'.format(epoch))
            break
    writer.close()


if __name__ == '__main__':
    main()
