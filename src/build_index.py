"""Lập danh sách cặp frame/mask từ ảnh mặt đã cắt (src/preprocess.py), chia theo split chính thức của FF++.

Chạy: python src/build_index.py
Kết quả: data/index/{train,val,test,dfd}.csv, mỗi dòng là một frame:
    frame, mask (rỗng với frame thật), method, video, label (0 thật, 1 giả)
Đường dẫn trong CSV là đường dẫn tương đối so với thư mục dữ liệu.
"""
import argparse
import csv
import json
import os

FF_FAKE = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
FIELDS = ['frame', 'mask', 'method', 'video', 'label']


def video_rows(root, seq, method, video, label):
    """Mọi frame của một video; frame giả kèm mask cùng tên."""
    base = seq
    frame_dir = os.path.join(root, base, 'frames', video)
    if not os.path.isdir(frame_dir):
        return [], 0
    rows, missing = [], 0
    for name in sorted(f for f in os.listdir(frame_dir) if f.endswith('.png')):
        mask = ''
        if label:
            mask = '{}/masks/{}/{}'.format(base, video, name)
            if not os.path.isfile(os.path.join(root, mask)):
                missing += 1
                continue
        rows.append({'frame': '{}/frames/{}/{}'.format(base, video, name),
                     'mask': mask, 'method': method, 'video': video,
                     'label': label})
    return rows, missing


def ff_split_rows(root, pairs):
    rows, missing, absent = [], 0, []
    for a, b in pairs:
        videos = [('original_sequences/youtube', 'original', v, 0)
                  for v in (a, b)]
        videos += [('manipulated_sequences/' + m, m, v, 1)
                   for m in FF_FAKE for v in (a + '_' + b, b + '_' + a)]
        for seq, method, video, label in videos:
            r, miss = video_rows(root, seq, method, video, label)
            if not r:
                absent.append(method + '/' + video)
            rows += r
            missing += miss
    return rows, missing, absent


def dfd_rows(root):
    rows, missing, absent = [], 0, []
    for seq, method, label in [
            ('manipulated_sequences/DeepFakeDetection', 'DFD', 1),
            ('original_sequences/actors', 'actors', 0)]:
        frame_root = os.path.join(root, seq, 'frames')
        for video in sorted(os.listdir(frame_root)):
            r, miss = video_rows(root, seq, method, video, label)
            if not r:
                absent.append(method + '/' + video)
            rows += r
            missing += miss
    return rows, missing, absent


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


def summarize(name, rows, missing, absent):
    counts = {}
    for r in rows:
        c = counts.setdefault(r['method'], [set(), 0])
        c[0].add(r['video'])
        c[1] += 1
    real = sum(1 for r in rows if not r['label'])
    print('{}: {} frames ({} real, {} fake), {} frames without mask skipped, '
          '{} videos absent'.format(name, len(rows), real, len(rows) - real,
                                    missing, len(absent)))
    for m, (videos, n) in counts.items():
        print('    {:<15} {:>5} videos {:>7} frames'.format(m, len(videos), n))
    if absent:
        print('    absent: ' + ', '.join(absent))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', default='data/faces')
    parser.add_argument('--splits', default='data/splits')
    parser.add_argument('--out', default='data/index')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for split in ('train', 'val', 'test'):
        with open(os.path.join(args.splits, split + '.json')) as f:
            pairs = json.load(f)
        rows, missing, absent = ff_split_rows(args.root, pairs)
        write_csv(os.path.join(args.out, split + '.csv'), rows)
        summarize(split, rows, missing, absent)

    rows, missing, absent = dfd_rows(args.root)
    write_csv(os.path.join(args.out, 'dfd.csv'), rows)
    summarize('dfd', rows, missing, absent)


if __name__ == '__main__':
    main()
