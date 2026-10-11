"""Xem tiến trình tiền xử lý (src/preprocess.py) theo từng nhóm, tự làm mới.

Chạy: python preprocess_status.py                 (c23, đọc data/faces/log.jsonl)
      python preprocess_status.py --out data/faces_c40
Ctrl+C để thoát.
"""
import argparse
import json
import os
import sys
import time

TOTAL = {
    'original_sequences/youtube': 1000,
    'original_sequences/actors': 363,
    'manipulated_sequences/Deepfakes': 1000,
    'manipulated_sequences/Face2Face': 1000,
    'manipulated_sequences/FaceSwap': 1000,
    'manipulated_sequences/NeuralTextures': 1000,
    'manipulated_sequences/DeepFakeDetection': 3068,
}
BAR_WIDTH = 25


def render(log_path, history):
    stats = {g: [0, 0, 0, 0] for g in TOTAL}
    if os.path.isfile(log_path):
        with open(log_path, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                s = stats.setdefault(r['group'], [0, 0, 0, 0])
                s[0] += 1
                s[1] += r['saved']
                s[2] += r['no_face']
                s[3] += r.get('empty_mask', 0)
    lines = ['Tien xu ly: ' + log_path, '',
             '{:<16} {:<27} {:>11} {:>8} {:>8} {:>6}'.format(
                 'nhom', '', 'video', 'frame', 'no_face', 'empty')]
    done_videos = 0
    for g, total in TOTAL.items():
        n, saved, no_face, empty = stats[g]
        done_videos += n
        filled = int(BAR_WIDTH * n / total)
        status = 'XONG' if n >= total else ('dang chay' if n else 'cho')
        lines.append('{:<16} [{}] {:>5}/{:<5} {:>8} {:>8} {:>6}  {}'.format(
            g.split('/')[-1][:16], '#' * filled + '-' * (BAR_WIDTH - filled),
            n, total, saved, no_face, empty, status))
    grand = sum(TOTAL.values())
    lines.append('')
    lines.append('TONG: {}/{} video ({:.1%})'.format(done_videos, grand,
                                                    done_videos / grand))
    now = time.time()
    history.append((now, done_videos))
    while len(history) > 2 and now - history[0][0] > 120:
        history.pop(0)
    t0, d0 = history[0]
    if done_videos > d0:
        rate = (done_videos - d0) / (now - t0)
        left = (grand - done_videos) / rate / 60
        lines.append('Toc do: {:.1f} video/phut - con lai khoang {}h{:02d}m'.format(
            rate * 60, int(left // 60), int(left % 60)))
    else:
        lines.append('Toc do: dang do...')
    lines.append('no_face = frame khong tim thay mat (bi bo qua); '
                 'empty = frame gia co mask trong')
    lines.append('Cap nhat: {}   (Ctrl+C de thoat)'.format(time.strftime('%H:%M:%S')))
    return '\n'.join(lines), done_videos >= grand


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='data/faces')
    parser.add_argument('--refresh', type=int, default=5)
    args = parser.parse_args()
    os.system('')  # bật mã ANSI trên Windows console
    history = []
    try:
        while True:
            text, finished = render(os.path.join(args.out, 'log.jsonl'), history)
            sys.stdout.write('\033[2J\033[H' + text + '\n')
            sys.stdout.flush()
            if finished:
                print('\nDa xu ly xong tat ca!')
                break
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
