"""Phân tích lỗi theo nhóm: IoU theo cặp diễn viên (DFD) và theo vị trí frame trong video.

Kết luận thật/giả cả video (--split video): video giả nếu tỉ lệ frame có vùng giả
(> 1% diện tích mặt) >= ngưỡng; ngưỡng chọn trên FF++ val, đo trên test và DFD.
Lưu <thư mục run>/video_rule.json cho demo.py và bảng runs/analysis_video.md.

Chạy: python src/analyze.py runs/unet_effb0/best.pt runs/deeplabv3plus_effb0/best.pt \
          runs/segformer_b0/best.pt --split dfd
Điểm từng frame (IoU ảnh giả, diện tích báo nhầm ảnh thật) lưu ở
<thư mục run>/analysis/scores_<split>.csv, chạy lại sẽ dùng file này thay vì chạy mô hình.
Bảng tổng hợp: runs/analysis_<split>.md
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

FA_AREA = 0.01  # giống metrics.py: ảnh thật bị báo nhầm khi vùng dự đoán > 1%


def frame_scores(checkpoint, split):
    path = os.path.join(os.path.dirname(checkpoint), 'analysis',
                        'scores_{}.csv'.format(split))
    if not os.path.exists(path) or 'area' not in open(path).readline():
        from dataset import FaceSegDataset, eval_transform
        from evaluate import load_checkpoint
        from visualize import rank_all
        model, cfg, _, device = load_checkpoint(checkpoint)
        ds = FaceSegDataset(os.path.join(cfg['index_dir'], split + '.csv'),
                            cfg['data_root'], eval_transform(cfg['image_size']))
        scores = rank_all(model, ds, cfg, device, cfg['threshold'])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['frame', 'method', 'video', 'label', 'score', 'area'])
            for i, method, label, s, a in scores:
                frame, _, _, video, _ = ds.rows[i]
                w.writerow([frame, method, video.split('/', 1)[1], label, s, a])
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['label'], r['score'], r['area'] = (int(r['label']), float(r['score']),
                                             float(r['area']))
    # vị trí frame trong 32 frame lấy mẫu của video (0 = frame đầu)
    by_video = defaultdict(list)
    for r in rows:
        by_video[(r['method'], r['video'])].append(r)
    for frames in by_video.values():
        frames.sort(key=lambda r: r['frame'])
        for k, r in enumerate(frames):
            r['pos'], r['last'] = k, k == len(frames) - 1
    return rows


def value(rows):
    """IoU trung bình (ảnh giả) hoặc tỉ lệ báo nhầm (ảnh thật)."""
    if rows[0]['label']:
        return np.mean([r['score'] for r in rows])
    return np.mean([r['score'] > FA_AREA for r in rows])


def table(head, lines):
    out = ['| ' + ' | '.join(head) + ' |', '|' + ' --- |' * len(head)]
    return out + ['| ' + ' | '.join(str(c) for c in l) + ' |' for l in lines]


def by_position(results, names):
    """Frame đầu, frame cuối và các frame giữa, theo từng phương pháp."""
    groups = [('frame đầu', lambda r: r['pos'] == 0),
              ('frame giữa', lambda r: r['pos'] > 0 and not r['last']),
              ('frame cuối', lambda r: r['last'])]
    methods = sorted({r['method'] for r in results[names[0]]})
    lines = []
    for m in methods:
        for g, keep in groups:
            cells = [m, g]
            for n in names:
                rs = [r for r in results[n] if r['method'] == m and keep(r)]
                v = value(rs)
                cells.append('{:.3f}'.format(v) if rs[0]['label']
                             else '{:.1%}'.format(v))
            lines.append(cells)
    return table(['Phương pháp', 'Vị trí'] + names, lines)


def by_pair(results, names, top=8):
    """DFD: IoU theo cặp diễn viên (đích_nguồn), sắp theo trung bình các mô hình."""
    pairs = defaultdict(lambda: defaultdict(list))
    for n in names:
        for r in results[n]:
            if r['label']:
                pairs[r['video'].split('__')[0]][n].append(r['score'])
    stats = []
    for p, d in pairs.items():
        ious = [np.mean(d[n]) for n in names]
        videos = len({r['video'] for r in results[names[0]]
                      if r['label'] and r['video'].startswith(p + '__')})
        stats.append((np.mean(ious), p, videos, ious))
    stats.sort()
    lines = [[p, v] + ['{:.3f}'.format(x) for x in ious]
             for _, p, v, ious in stats[:top] + stats[-top:]]
    lines.insert(top, ['…'] * (2 + len(names)))
    means = np.array([s[0] for s in stats])
    # mỗi cặp có trọng số như nhau, khác IoU theo frame trong evaluate.py
    summary = ['{} cặp diễn viên. IoU trung bình theo cặp: trung vị {:.3f}, '
               '{} cặp dưới 0,3, {} cặp trên 0,8.'.format(
                   len(stats), np.median(means), (means < 0.3).sum(),
                   (means > 0.8).sum())]
    return summary + [''] + table(['Cặp', 'Số video'] + names, lines), stats


def video_scores(rows):
    """Mỗi video: tỉ lệ frame có vùng giả dự đoán > 1% diện tích mặt, và nhãn."""
    flags, labels = defaultdict(list), {}
    for r in rows:
        key = (r['method'], r['video'])
        flags[key].append(r['area'] > FA_AREA)
        labels[key] = r['label']
    keys = sorted(flags)
    return (np.array([np.mean(flags[k]) for k in keys]),
            np.array([labels[k] for k in keys]))


def rule_stats(score, label, t):
    pred = score >= t
    real, fake = pred[label == 0], pred[label == 1]
    return {'real_videos': int(len(real)), 'fake_videos': int(len(fake)),
            'real_correct': float(1 - real.mean()), 'fake_correct': float(fake.mean()),
            'balanced_acc': float((1 - real.mean() + fake.mean()) / 2),
            'auc': float(roc_auc_score(label, score))}


def video_rule(checkpoint):
    """Chọn ngưỡng trên val theo độ chính xác cân bằng (val có 4 video giả mỗi
    video thật), rồi đo trên test và DFD."""
    val_s, val_y = video_scores(frame_scores(checkpoint, 'val'))
    grid = np.arange(1, 33) / 32   # 32 frame mỗi video
    t = max(grid, key=lambda g: rule_stats(val_s, val_y, g)['balanced_acc'])
    rule = {'threshold': float(t), 'fa_area': FA_AREA}
    for split in ('val', 'test', 'dfd'):
        rule[split] = rule_stats(*video_scores(frame_scores(checkpoint, split)), t)
    with open(os.path.join(os.path.dirname(checkpoint), 'video_rule.json'), 'w',
              encoding='utf-8') as f:
        json.dump(rule, f, indent=2)
    return rule


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('checkpoints', nargs='+')
    parser.add_argument('--split', default='dfd',
                        choices=['val', 'test', 'dfd', 'video'])
    args = parser.parse_args()

    names = [os.path.basename(os.path.dirname(c)) for c in args.checkpoints]
    if args.split == 'video':
        lines = ['# Kết luận thật/giả theo video', '',
                 'Video giả nếu tỉ lệ frame có vùng giả (> 1% diện tích mặt) >= ngưỡng; '
                 'ngưỡng chọn trên FF++ val.', '']
        rows = []
        for n, c in zip(names, args.checkpoints):
            rule = video_rule(c)
            for split in ('val', 'test', 'dfd'):
                s = rule[split]
                rows.append([n, '{:.0%}'.format(rule['threshold']), split,
                             '{} / {}'.format(s['real_videos'], s['fake_videos']),
                             '{:.1%}'.format(s['real_correct']),
                             '{:.1%}'.format(s['fake_correct']),
                             '{:.1%}'.format(s['balanced_acc']), '{:.3f}'.format(s['auc'])])
        lines += table(['Mô hình', 'Ngưỡng', 'Tập', 'Video thật / giả', 'Đúng video thật',
                        'Đúng video giả', 'Chính xác cân bằng', 'AUC'], rows)
    else:
        results = {n: frame_scores(c, args.split)
                   for n, c in zip(names, args.checkpoints)}
        lines = ['# Phân tích lỗi trên {}'.format(args.split), '',
                 '## IoU (ảnh giả) và báo nhầm (ảnh thật) theo vị trí frame', '']
        lines += by_position(results, names)
        if args.split == 'dfd':
            pair_lines, _ = by_pair(results, names)
            lines += ['', '## DFD: IoU theo cặp diễn viên (8 cặp kém nhất, '
                      '8 cặp tốt nhất)', ''] + pair_lines
    out = os.path.join('runs', 'analysis_{}.md'.format(args.split))
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('saved', out)


if __name__ == '__main__':
    main()
