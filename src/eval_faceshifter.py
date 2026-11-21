"""Thử chéo kiểu giả mạo chưa thấy: FaceShifter (FF++), 140 video giả của tập test FF++,
ghép với 140 video thật cùng tập test (data/clips_ff). Chỉ đo phân loại (không dùng mask).

1. python src/eval_faceshifter.py prepare
     → liên kết cứng video test vào data/FaceShifter_test/manipulated_sequences/FaceShifter/{c23,c40}/videos
2. python src/preprocess.py --src data/FaceShifter_test --out data/clips_faceshifter --clip-len 8 --follow-face --groups FaceShifter
   python src/preprocess.py --src data/FaceShifter_test --compression c40 --out data/clips_faceshifter_c40 --boxes-from data/clips_faceshifter --workers 4 --groups FaceShifter
3. python src/eval_faceshifter.py eval runs/b1_dfd/best.pt [runs/b1_mix/best.pt ...]
     → data/index_faceshifter[_c40]/faceshifter.csv, <thư mục run>/eval_faceshifter[_c40].json
"""
import argparse
import csv
import json
import os
import sys

SRC, DST = 'data/FaceShifter', 'data/FaceShifter_test'
GROUP = 'manipulated_sequences/FaceShifter'


def test_videos():
    with open('data/splits/test.json', encoding='utf-8') as f:
        return [v for a, b in json.load(f) for v in (f'{a}_{b}', f'{b}_{a}')]


def prepare():
    for comp in ('c23', 'c40'):
        d = os.path.join(DST, GROUP, comp, 'videos')
        os.makedirs(d, exist_ok=True)
        for v in test_videos():
            dst = os.path.join(d, v + '.mp4')
            if not os.path.exists(dst):
                os.link(os.path.join(SRC, GROUP, comp, 'videos', v + '.mp4'), dst)
    print(len(test_videos()), 'video giả mỗi mức nén')


def build_index(comp):
    sfx = '' if comp == 'c23' else '_' + comp
    with open(f'data/index_clips_ff{sfx}/test.csv', newline='', encoding='utf-8') as f:
        rows = [dict(r, frame=f'clips_ff{sfx}/' + r['frame']) for r in csv.DictReader(f)
                if r['method'] == 'original']
    real_n = len({r['video'] for r in rows})
    root = f'clips_faceshifter{sfx}'
    for video in test_videos():
        d = os.path.join('data', root, GROUP, 'frames', video)
        with open(os.path.join(d, 'boxes.json'), encoding='utf-8') as f:
            boxes = json.load(f)
        rows += [{'frame': f'{root}/{GROUP}/frames/{video}/{name}', 'mask': '',
                  'method': 'FaceShifter', 'video': video, 'label': 1, 'compression': comp,
                  'bbox': ','.join(map(str, boxes[name]))}
                 for name in sorted(os.listdir(d)) if name.endswith('.png')]
    assert real_n == 140, real_n
    index = f'data/index_faceshifter{sfx}'
    os.makedirs(index, exist_ok=True)
    with open(os.path.join(index, 'faceshifter.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return index


def evaluate(ckpts):
    from evaluate import eval_loader, load_checkpoint, run_eval
    for comp in ('c23', 'c40'):
        index = build_index(comp)
        for ckpt in ckpts:
            model, cfg, meta, device = load_checkpoint(ckpt, comp)
            cfg = dict(cfg, data_root='data', index_dir=index)   # đường dẫn ảnh trong csv tính từ data/
            summary = run_eval(model, eval_loader(cfg, 'faceshifter'), device, cfg['threshold'],
                               desc=f'faceshifter {comp}')
            run_dir = os.path.dirname(ckpt)
            name = 'eval_faceshifter' + ('' if comp == 'c23' else '_' + comp) + '.json'
            with open(os.path.join(run_dir, name), 'w', encoding='utf-8') as f:
                json.dump({'epoch': meta['epoch'], 'split': 'faceshifter', 'compression': comp,
                           'cascade': False, 'fraction': 1.0, 'metrics': summary}, f, indent=2)
            v = summary['cls']['video']
            print('{} {} (epoch {}): {} video, acc cân bằng {:.1%}, AUC {:.3f}, bắt FaceShifter '
                  '{:.1%}, nhận đúng video thật {:.1%}'.format(
                      run_dir, comp, meta['epoch'], v['n'],
                      (v['bin_recall'] + v['bin_real_recall']) / 2, v['bin_auc'],
                      v['bin_recall'], v['bin_real_recall']))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('step', choices=['prepare', 'eval'])
    ap.add_argument('ckpts', nargs='*')
    a = ap.parse_args()
    prepare() if a.step == 'prepare' else evaluate(a.ckpts)


if __name__ == '__main__':
    sys.exit(main())
