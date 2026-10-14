"""Mô hình 2: cắt mặt video TalkingHeadBench và tạo danh sách mẫu thật / do AI tạo.

Chạy: python src/preprocess_ai.py
Video giả (AI tạo): data/TalkingHeadBench/fake/<bộ sinh>/<split>/*.mp4, cắt mặt giống hệt
Mô hình 1 (preprocess.process_video: RetinaFace, nới 1,3 lần, 256×256), 16 frame/video,
lưu ở data/faces_ai/. additional_dataset (Hallo3, MAGI-1) chỉ dùng để test (split 'extra').
Video thật: FF++ youtube theo real_dataset_split_official_ff++.json (split chính thức FF++,
không lẫn với tập test của Mô hình 1); ảnh mặt đã có sẵn trong data/faces.
Kết quả: data/index_ai/{train,val,test,extra}.csv, cột frame (tương đối với data/),
source, video, label (1 = do AI tạo).
"""
import argparse
import csv
import glob
import json
import multiprocessing as mp
import os

from tqdm import tqdm

from preprocess import init_worker, process_video

SPLITS = {'train': 'Train', 'val': 'Val', 'test': 'Test'}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--src', default='data/TalkingHeadBench')
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()

    out = 'data/faces_ai'
    log_path = os.path.join(out, 'log.jsonl')
    os.makedirs(out, exist_ok=True)
    done = set()
    if os.path.isfile(log_path):
        with open(log_path, encoding='utf-8') as f:
            done = {(r['group'], r['video']) for r in map(json.loads, f)}
    videos = sorted(glob.glob(os.path.join(args.src, 'fake', '*', '*', '*.mp4')))
    tasks = []
    for v in videos:
        group = os.path.relpath(os.path.dirname(v), args.src).replace('\\', '/')
        if (group, os.path.splitext(os.path.basename(v))[0]) not in done:
            tasks.append((group, v, None, out, args.frames, 1.3, 256))
    print('{} videos to process ({} done)'.format(len(tasks), len(done)))
    if tasks:
        with mp.Pool(args.workers, initializer=init_worker, initargs=(0,)) as pool, \
                open(log_path, 'a', encoding='utf-8') as log:
            for r in tqdm(pool.imap_unordered(process_video, tasks), total=len(tasks)):
                log.write(json.dumps(r) + '\n')
                log.flush()

    rows = {s: [] for s in ('train', 'val', 'test', 'extra')}
    for d in sorted(glob.glob(os.path.join(out, 'fake', '*', '*', 'frames', '*'))):
        parts = os.path.relpath(d, out).replace('\\', '/').split('/')
        # fake/<bộ sinh>/<split>/frames/<video> hoặc fake/additional_dataset/<bộ sinh>/...
        split = 'extra' if parts[1] == 'additional_dataset' else parts[2]
        source = parts[2] if split == 'extra' else parts[1]
        for f in sorted(os.listdir(d)):
            rows[split].append(('faces_ai/' + '/'.join(parts) + '/' + f, source, parts[-1], 1))
    with open(os.path.join(args.src, 'real', 'real_dataset_split_official_ff++.json'),
              encoding='utf-8') as f:
        real = json.load(f)
    for split, key in SPLITS.items():
        for name in real[key]['FaceForensics++']:
            video = os.path.splitext(name)[0]
            rel = 'faces/original_sequences/youtube/frames/' + video
            for f in sorted(os.listdir(os.path.join('data', rel))):
                rows[split].append((rel + '/' + f, 'real_ff++', video, 0))

    os.makedirs('data/index_ai', exist_ok=True)
    for split, rs in rows.items():
        with open(os.path.join('data/index_ai', split + '.csv'), 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['frame', 'source', 'video', 'label'])
            w.writerows(rs)
        print(split, len(rs), 'frames,', sum(r[3] for r in rs), 'AI,',
              len({r[2] for r in rs if r[3]}), 'AI videos,',
              len({r[2] for r in rs if not r[3]}), 'real videos')


if __name__ == '__main__':
    main()
