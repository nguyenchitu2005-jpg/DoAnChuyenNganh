"""Thử chéo trên Celeb-DF v2 (518 video của List_of_testing_videos.txt), chỉ đo phân loại
(Celeb-DF không có mask). Ảnh mặt cắt giống FF++ của b1_dfd (data/clips_ff).

1. python src/eval_celebdf.py prepare "C:/.../Celeb-DF-v2"
     → chép video test vào data/CelebDF/celebdf/{real,fake}/c23/videos
2. python src/preprocess.py --src data/CelebDF --out data/clips_celebdf --clip-len 8 --follow-face --groups celebdf
3. python src/eval_celebdf.py eval runs/b1_dfd/best.pt [runs/b1_mix/best.pt ...]
     → data/index_celebdf/celebdf.csv, <thư mục run>/eval_celebdf.json
Nhãn trong danh sách Celeb-DF: 1 = thật, 0 = giả (ngược với dự án: 1 = giả).
"""
import argparse
import csv
import json
import os
import shutil
import sys

SRC, ROOT, INDEX = 'data/CelebDF', 'data/clips_celebdf', 'data/index_celebdf'
GROUPS = [('celebdf/real', 'original', 0), ('celebdf/fake', 'CelebDF', 1)]


def prepare(celebdf_dir):
    with open(os.path.join(celebdf_dir, 'List_of_testing_videos.txt'), encoding='utf-8') as f:
        items = [line.split() for line in f if line.strip()]
    for lab, rel in items:
        fake = rel.startswith('Celeb-synthesis/')
        assert fake == (lab == '0'), f'nhãn không khớp thư mục: {lab} {rel}'
        dst = os.path.join(SRC, 'celebdf', 'fake' if fake else 'real', 'c23', 'videos',
                           os.path.basename(rel))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(celebdf_dir, rel), dst)
    print(len(items), 'video:', sum(lab == '1' for lab, _ in items), 'thật,',
          sum(lab == '0' for lab, _ in items), 'giả')


def build_index():
    rows = []
    for group, method, label in GROUPS:
        frame_root = os.path.join(ROOT, group, 'frames')
        for video in sorted(os.listdir(frame_root)):
            d = os.path.join(frame_root, video)
            with open(os.path.join(d, 'boxes.json'), encoding='utf-8') as f:
                boxes = json.load(f)
            rows += [{'frame': f'{group}/frames/{video}/{name}', 'mask': '', 'method': method,
                      'video': video, 'label': label, 'compression': 'c23',
                      'bbox': ','.join(map(str, boxes[name]))}
                     for name in sorted(os.listdir(d)) if name.endswith('.png')]
    os.makedirs(INDEX, exist_ok=True)
    with open(os.path.join(INDEX, 'celebdf.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), 'khung,', len({(r['label'], r['video']) for r in rows}), 'video')


def evaluate(ckpts):
    from evaluate import eval_loader, load_checkpoint, run_eval
    if not os.path.isfile(os.path.join(INDEX, 'celebdf.csv')):
        build_index()
    for ckpt in ckpts:
        model, cfg, meta, device = load_checkpoint(ckpt)
        cfg = dict(cfg, data_root=ROOT, index_dir=INDEX)
        summary = run_eval(model, eval_loader(cfg, 'celebdf'), device, cfg['threshold'],
                           desc='celebdf')
        run_dir = os.path.dirname(ckpt)
        with open(os.path.join(run_dir, 'eval_celebdf.json'), 'w', encoding='utf-8') as f:
            json.dump({'epoch': meta['epoch'], 'split': 'celebdf', 'compression': 'c23',
                       'cascade': False, 'fraction': 1.0, 'metrics': summary}, f, indent=2)
        v = summary['cls']['video']
        print('{} (epoch {}): {} video, acc cân bằng {:.1%}, AUC {:.3f}, bắt video giả {:.1%}, '
              'nhận đúng video thật {:.1%}'.format(
                  run_dir, meta['epoch'], v['n'], (v['bin_recall'] + v['bin_real_recall']) / 2,
                  v['bin_auc'], v['bin_recall'], v['bin_real_recall']))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('step', choices=['prepare', 'eval'])
    ap.add_argument('paths', nargs='+', help='prepare: thư mục Celeb-DF v2; eval: checkpoint')
    a = ap.parse_args()
    prepare(a.paths[0]) if a.step == 'prepare' else evaluate(a.paths)


if __name__ == '__main__':
    sys.exit(main())
