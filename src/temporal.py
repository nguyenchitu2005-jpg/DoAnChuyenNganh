"""T1: mô hình thời gian trên đặc trưng đã tính sẵn của encoder B1 đóng băng (extract_features.py)
và điểm mốc MediaPipe (face_landmarks.py --points). Train được trên GPU nhỏ trong vài phút.

Chạy:  python src/temporal.py --name t1_full --roi --landmarks
Ablation (kế hoạch, bước 3):
  --no-temporal              : MLP từng khung rồi lấy trung bình (không trộn thông tin giữa các khung)
  (mặc định)                 : + Transformer thời gian trên đặc trưng GAP của Swin
  --landmarks                : + điểm mốc chuẩn hoá và vận tốc điểm mốc giữa hai khung liền kề
  --roi                      : + đặc trưng vùng mắt trái, mắt phải, mũi, miệng
  --dfd-train                : thêm phần train của DFD (14 diễn viên, chia theo diễn viên, xem dfd_part);
                               đo trên 14 diễn viên còn lại ("dfdtest") ở mọi lần chạy để so được
  --zoom                     : + đặc trưng mức zoom 2 (mặt sát) và 3 (vùng miệng), cùng Swin
  --zoom-hires               : như --zoom, vùng miệng cắt từ khung hình gốc (độ phân giải cao)
  --dfd-max-yaw θ (mặc định 30): chỉ với DFD (train lẫn test) bỏ đoạn có |yaw| trung bình ≥ θ;
                               video không còn đoạn nào giữ đoạn ít nghiêng nhất. FF++ không lọc.
                               --dfd-max-yaw 999 để tắt.
  --dfd-max-bad r (mặc định 1 = tắt): chỉ với DFD bỏ đoạn có tỉ lệ khung xấu ≥ r (MediaPipe không
                               thấy mặt hoặc mặt trôi ra mép), áp cho cả thật lẫn giả, train lẫn test.
Dòng "B1 trung bình khung" (mốc so sánh) luôn được in, tính từ xác suất B1 trên cùng các khung.
Train trộn c23 + c40 như B1; chọn epoch theo accuracy cân bằng mức video trung bình val c23/c40.
Kết quả: runs/temporal/<name>/{best.pt, history.jsonl, results.json}
"""
import argparse
import csv
import json
import os
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from dataset import METHODS

CLIP, STRIDE = 8, 1   # số khung mỗi đoạn, khoảng cách khung trong đoạn (đặt lại từ --clip-len/--clip-stride)
N_CLS = 5
DFD_TRAIN_ACTORS = 14   # chia DFD theo diễn viên (seed 0): 14 diễn viên train, 14 test


def dfd_part(video, seed=0):
    """'train', 'test' hoặc None cho một đoạn DFD. Video giả ghép mặt giữa diễn viên train và
    diễn viên test bị bỏ (chứa danh tính của cả hai phần → rò rỉ)."""
    import random
    actors = [f'{k:02d}' for k in range(1, 29)]
    random.Random(seed).shuffle(actors)
    part = {a: 'train' if k < DFD_TRAIN_ACTORS else 'test' for k, a in enumerate(actors)}
    ids = video.split('/')[-1].split('__')[0].split('_')    # '01' (thật) hoặc '01_02' (giả)
    parts = {part[i] for i in ids}
    return parts.pop() if len(parts) == 1 else None


def load_split(feat_dir, index_dir, split, suffix, lm):
    """Các đoạn của một tập: list dict(video, label, method, idx (CLIP chỉ số khung), yaw)."""
    d = np.load(os.path.join(feat_dir, f'{split}{suffix}.npz'))
    with open(os.path.join(index_dir + suffix, split + '.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert [r['frame'] for r in rows] == list(d['frames']), split + suffix
    by_video = defaultdict(list)
    for i, r in enumerate(rows):
        frame_no = int(os.path.splitext(os.path.basename(r['frame']))[0])
        by_video[r['method'] + '/' + r['video']].append((frame_no, i, r))
    clips = []
    for v, items in by_video.items():
        items.sort()
        groups, cur = [], [items[0]]
        for it in items[1:]:   # cùng đoạn = trong độ dài một đoạn tính từ khung đầu (có thể khuyết)
            if it[0] - cur[0][0] < CLIP * STRIDE:
                cur.append(it)
            else:
                groups.append(cur)
                cur = [it]
        groups.append(cur)
        for g in groups:
            if len(g) < CLIP // 2:              # shortcut: đoạn bị khuyết quá nửa (mask rỗng) bỏ
                continue
            # khung bị khuyết (không thấy mặt / mask rỗng): lấy khung có mặt gần nhất, giữ thứ tự
            nums = [n for n, _, _ in g]
            idx = [g[min(range(len(g)), key=lambda k: (abs(nums[k] - s), nums[k] > s))][1]
                   for s in range(nums[0], nums[0] + CLIP * STRIDE, STRIDE)]
            r = g[0][2]
            yaws = [lm['yaw'].get(rows[i]['frame']) for i in idx]
            yaws = [abs(y) for y in yaws if y is not None]
            clips.append(dict(video=suffix + v, label=int(r['label']),
                              method=METHODS.index(r['method']), idx=idx,
                              yaw=np.mean(yaws) if yaws else 999.0,
                              bad=np.mean([rows[i]['frame'] in lm['bad'] for i in idx])))
    feats = {'gap': torch.from_numpy(d['gap']), 'roi': torch.from_numpy(d['roi']),
             'prob': torch.from_numpy(d['prob'])}
    pts = lm['points'](list(d['frames']))
    return clips, feats, pts


def keep_clips(clips, key, limit):
    """Giữ đoạn có c[key] < limit; video không còn đoạn nào giữ đoạn có c[key] nhỏ nhất
    (mọi video vẫn có kết luận, số video đánh giá không đổi)."""
    per = defaultdict(list)
    for c in clips:
        per[c['video']].append(c)
    out = []
    for cs in per.values():
        keep = [c for c in cs if c[key] < limit]
        out += keep or [min(cs, key=lambda c: c[key])]
    return out


def landmark_table(index_dir, size=224, max_off=0.2):
    """yaw theo frame, tập khung "xấu" và hàm lấy điểm mốc chuẩn hoá (N, 114, 2) cho danh sách frame.
    Khung xấu = MediaPipe không thấy mặt, hoặc tâm mũi lệch khỏi giữa ảnh > max_off·size
    (mặt trôi ra mép do cả đoạn dùng chung một khung cắt)."""
    with open(os.path.join(index_dir, 'landmarks.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    yaw = {r['frame']: float(r['yaw']) for r in rows if r['yaw']}
    bad = {r['frame'] for r in rows if not r['yaw'] or max(
        abs((float(r['nose_x1']) + float(r['nose_x2'])) / 2 - size / 2),
        abs((float(r['nose_y1']) + float(r['nose_y2'])) / 2 - size / 2)) > max_off * size}
    z = np.load(os.path.join(index_dir, 'landmarks.npz'))
    pos = {f: i for i, f in enumerate(z['frames'])}
    pts_all = z['points'][..., :2]

    def points(frames):
        p = np.stack([pts_all[pos[f]] if f in pos else np.full(pts_all.shape[1:], np.nan,
                                                                np.float32) for f in frames])
        # chuẩn hoá từng khung: trừ tâm, chia độ lệch chuẩn → bất biến vị trí và cỡ mặt
        with warnings.catch_warnings():   # khung MediaPipe không thấy mặt: toàn NaN → về 0
            warnings.simplefilter('ignore', RuntimeWarning)
            c = np.nanmean(p, 1, keepdims=True)
            s = np.nanstd(p - c, axis=(1, 2), keepdims=True) + 1e-6
        return torch.from_numpy(np.nan_to_num((p - c) / s)).float()
    return {'yaw': yaw, 'bad': bad, 'points': points}


class T1(nn.Module):
    def __init__(self, use_roi, use_lm, temporal, levels=(0,), dim=256, n_points=114):
        super().__init__()
        self.use_roi, self.use_lm, self.temporal = use_roi, use_lm, temporal
        self.levels = list(levels)   # mức zoom dùng (chỉ số trong gap), xem --zoom / --zoom-hires
        n = len(self.levels)
        self.gap = nn.Sequential(nn.LayerNorm(n * 768), nn.Linear(n * 768, dim))
        if use_roi:
            self.roi = nn.Sequential(nn.LayerNorm(4 * 576), nn.Linear(4 * 576, dim))
        if use_lm:   # vị trí + vận tốc (so với khung trước) của từng điểm mốc
            self.lm = nn.Sequential(nn.Linear(4 * n_points, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pos = nn.Parameter(torch.zeros(1, CLIP, dim))
        if temporal:
            layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, dropout=0.1, batch_first=True,
                                               norm_first=True)
            self.mix = nn.TransformerEncoder(layer, 2)
        else:
            self.mix = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.drop = nn.Dropout(0.2)
        self.bin_head = nn.Linear(dim, 1)
        self.cls_head = nn.Linear(dim, N_CLS)
        self.frame_head = nn.Linear(dim, 1)   # thật/giả từng khung, sau khi đã trộn với khung lân cận

    def forward(self, gap, roi=None, pts=None):          # gap (B, CLIP, số mức, 768)
        h = self.gap(gap[:, :, self.levels].flatten(2))
        if self.use_roi:
            h = h + self.roi(roi)
        if self.use_lm:
            vel = torch.diff(pts, dim=1, prepend=pts[:, :1])
            h = h + self.lm(torch.cat([pts, vel], -1).flatten(2))
        tokens = self.drop(self.mix(h + self.pos if self.temporal else h))   # (B, CLIP, dim)
        h = tokens.mean(1)
        return self.bin_head(h)[:, 0], self.cls_head(h), self.frame_head(tokens)[..., 0]


def batch(clips, feats, pts, ids, device):
    idx = torch.tensor([clips[i]['idx'] for i in ids])
    return (feats['gap'][idx].float().to(device), feats['roi'][idx].float().to(device),
            pts[idx].flatten(2).to(device),
            torch.tensor([clips[i]['label'] for i in ids], dtype=torch.float, device=device),
            torch.tensor([clips[i]['method'] for i in ids], device=device))


@torch.no_grad()
def predict(model, data, device, bs=1024, frames=False):
    """Xác suất giả mỗi đoạn; frames=True trả thêm (số đoạn, CLIP) xác suất từng khung."""
    clips, feats, pts = data
    model.eval()
    out = []
    for k in range(0, len(clips), bs):
        g, r, p, *_ = batch(clips, feats, pts, range(k, min(k + bs, len(clips))), device)
        b, _, fr = model(g, r, p)
        out.append(torch.cat([torch.sigmoid(b)[:, None], torch.sigmoid(fr)], 1).cpu())
    out = torch.cat(out).numpy()
    return (out[:, 0], out[:, 1:]) if frames else out[:, 0]


def video_metrics(clips, clip_prob):
    """Mức video (= trung bình các đoạn): accuracy cân bằng (ngưỡng 0,5), AUC, và tỉ lệ nhận
    đúng theo từng phương pháp ('original'/'actors' = video thật, còn lại = bắt được video giả)."""
    per = defaultdict(list)
    for c, p in zip(clips, clip_prob):
        per[c['video']].append((p, c['label'], c['method']))
    p = np.array([np.mean([x[0] for x in items]) for items in per.values()])
    y = np.array([items[0][1] for items in per.values()])
    m = np.array([items[0][2] for items in per.values()])
    pred = p > 0.5
    by_method = {METHODS[k]: float((pred[m == k] == bool(y[m == k][0])).mean()) for k in set(m)}
    return {'bal_acc': float(((~pred[y == 0]).mean() + pred[y == 1].mean()) / 2),
            'auc': float(roc_auc_score(y, p)), 'n_video': len(y), 'by_method': by_method}


def frame_metrics(clips, frame_prob):
    """Mức khung: accuracy cân bằng (0,5) và AUC; mỗi khung tính một lần (bỏ khung lặp để đủ đoạn)."""
    p, y = [], []
    for c, fp in zip(clips, frame_prob):
        n = len(dict.fromkeys(c['idx']))
        p += list(fp[:n])
        y += [c['label']] * n
    p, y = np.array(p), np.array(y)
    pred = p > 0.5
    return {'bal_acc': float(((~pred[y == 0]).mean() + pred[y == 1].mean()) / 2),
            'auc': float(roc_auc_score(y, p))}


def frame_mean_prob(clips, feats):
    """Mốc so sánh: xác suất B1 trung bình các khung của đoạn (= cách B1 kết luận)."""
    return np.array([feats['prob'][c['idx']].mean().item() for c in clips])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--features', default='runs/b1_mix/features')
    ap.add_argument('--index', default='data/index_clips')
    ap.add_argument('--roi', action='store_true')
    ap.add_argument('--landmarks', action='store_true')
    ap.add_argument('--no-temporal', dest='temporal', action='store_false')
    ap.add_argument('--zoom', action='store_true')
    ap.add_argument('--zoom-hires', dest='zoom_hires', action='store_true',
                    help='như --zoom nhưng mức 3 là vùng miệng cắt từ khung hình gốc (mức 4 trong '
                         'đặc trưng, extract_features.py --mouth-root)')
    ap.add_argument('--dfd-max-yaw', dest='dfd_max_yaw', type=float, default=30)
    ap.add_argument('--dfd-max-bad', dest='dfd_max_bad', type=float, default=1,
                    help='DFD: bỏ đoạn có tỉ lệ khung xấu (không thấy mặt / mặt lệch ra mép) '
                         '≥ ngưỡng, ví dụ 0.5; 1 = không lọc')
    ap.add_argument('--dfd-train', dest='dfd_train', action='store_true',
                    help='thêm video của 14 diễn viên DFD (phần train) vào tập train')
    ap.add_argument('--clip-len', dest='clip_len', type=int, default=8,
                    help='phải khớp lúc cắt (preprocess.py --clip-len)')
    ap.add_argument('--clip-stride', dest='clip_stride', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch-size', dest='bs', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    global CLIP, STRIDE
    CLIP, STRIDE = args.clip_len, args.clip_stride
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm = landmark_table(args.index)
    sets = {f'{s}{x}': load_split(args.features, args.index, s, x, lm)
            for s in ('train', 'val', 'test', 'dfd') for x in ('', '_c40')}
    for x in ('', '_c40'):   # 14 diễn viên DFD chưa thấy: tập test chung cho mọi lần chạy
        c, f, p = sets['dfd' + x]
        n_vid = len({k['video'] for k in c})
        if args.dfd_max_bad < 1:
            c = keep_clips(c, 'bad', args.dfd_max_bad)
        c = keep_clips(c, 'yaw', args.dfd_max_yaw)
        assert len({k['video'] for k in c}) == n_vid
        print(f'dfd{x}: giữ {len(c)} đoạn sau lọc khung xấu / góc nghiêng')
        sets['dfd' + x] = (c, f, p)
        sets['dfdtest' + x] = ([k for k in c if dfd_part(k['video']) == 'test'], f, p)
        sets['dfdtrain' + x] = ([k for k in c if dfd_part(k['video']) == 'train'], f, p)

    # train trộn c23 + c40 (+ phần train của DFD nếu --dfd-train): ghép các bộ đoạn,
    # chỉ số khung của mỗi bộ dời theo tổng số khung các bộ trước
    parts = ['train', 'train_c40'] + (['dfdtrain', 'dfdtrain_c40'] if args.dfd_train else [])
    clips, feats, pts, off = [], defaultdict(list), [], 0
    for name in parts:
        c, f, p = sets[name]
        clips += [dict(k, idx=[i + off for i in k['idx']]) for k in c]
        for key in f:
            feats[key].append(f[key])
        pts.append(p)
        off += len(f['gap'])
    feats = {k: torch.cat(v) for k, v in feats.items()}
    pts = torch.cat(pts)
    labels = np.array([c['label'] for c in clips])
    w = np.where(labels == 1, 0.5 / labels.sum(), 0.5 / (len(labels) - labels.sum()))
    sampler = torch.utils.data.WeightedRandomSampler(w, len(clips), replacement=True)

    levels = [0, 1, 3] if args.zoom_hires else [0, 1, 2] if args.zoom else [0]
    model = T1(args.roi, args.landmarks, args.temporal, levels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * (len(clips) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    run_dir = os.path.join('runs', 'temporal', args.name)
    os.makedirs(run_dir, exist_ok=True)
    best, hist = -1, open(os.path.join(run_dir, 'history.jsonl'), 'w', encoding='utf-8')
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(sampler)
        tot = 0.0
        for k in range(0, len(order) - args.bs + 1, args.bs):
            g, r, p, y, m = batch(clips, feats, pts, order[k:k + args.bs], device)
            b, c, fr = model(g, r, p)
            ff = m < N_CLS   # đầu 5 lớp chỉ cho ảnh FF++
            # nhãn từng khung = nhãn video (đã kiểm: ~100% khung của video giả có mask khác rỗng)
            loss = (F.binary_cross_entropy_with_logits(b, y) + 0.5 * F.cross_entropy(c[ff], m[ff])
                    + 0.5 * F.binary_cross_entropy_with_logits(fr, y[:, None].expand_as(fr)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        val = {t: video_metrics(sets[t][0], predict(model, sets[t], device))
               for t in ('val', 'val_c40')}
        score = (val['val']['bal_acc'] + val['val_c40']['bal_acc']) / 2
        hist.write(json.dumps({'epoch': epoch, 'loss': tot / (len(order) // args.bs),
                               'val': val, 'score': score}) + '\n')
        hist.flush()
        print(f'epoch {epoch:2d} loss {tot / (len(order) // args.bs):.4f} '
              f'val bal_acc c23 {val["val"]["bal_acc"]:.3f} c40 {val["val_c40"]["bal_acc"]:.3f}')
        if score > best:
            best = score
            torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': epoch},
                       os.path.join(run_dir, 'best.pt'))

    model.load_state_dict(torch.load(os.path.join(run_dir, 'best.pt'))['model'])
    res = {}
    print(f'\n{args.name} (epoch tốt nhất trên val) — acc cân bằng / AUC mức video')
    tags = ['test', 'test_c40', 'dfdtest', 'dfdtest_c40']
    if not args.dfd_train:   # cả DFD chỉ là thử chéo khi không train trên phần nào của DFD
        tags += ['dfd', 'dfd_c40']
    for t in tags:
        clips_t, feats_t, _ = sets[t]
        clip_p, frame_p = predict(model, sets[t], device, frames=True)
        res[t] = {'T1': video_metrics(clips_t, clip_p),
                  'B1_frame_mean': video_metrics(clips_t, frame_mean_prob(clips_t, feats_t)),
                  'T1_frame': frame_metrics(clips_t, frame_p),
                  'B1_frame': frame_metrics(clips_t, [feats_t['prob'][c['idx']].numpy()
                                                      for c in clips_t])}
        a, b = res[t]['T1'], res[t]['B1_frame_mean']
        print(f'{t:<12} T1 {a["bal_acc"]:.1%} / {a["auc"]:.3f}   '
              f'B1 trung bình khung {b["bal_acc"]:.1%} / {b["auc"]:.3f}')
        print(' ' * 13 + '  '.join(f'{k} {a["by_method"][k]:.0%} (B1 {b["by_method"][k]:.0%})'
                                   for k in sorted(a['by_method'])))
        fa, fb = res[t]['T1_frame'], res[t]['B1_frame']
        print(' ' * 13 + f'từng khung: T1 {fa["bal_acc"]:.1%} / {fa["auc"]:.3f}   '
              f'B1 {fb["bal_acc"]:.1%} / {fb["auc"]:.3f}')
    with open(os.path.join(run_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump({'args': vars(args), 'results': res}, f, indent=1)


if __name__ == '__main__':
    main()
