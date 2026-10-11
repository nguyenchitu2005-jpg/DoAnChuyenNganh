"""Hiển thị tiến trình tải FaceForensics++ trực tiếp trên terminal.

Chạy: python watch_progress.py   (Ctrl+C để thoát)
"""
import os
import sys
import time

ROOT = os.path.join('data', 'FaceForensics')
FF = 'manipulated_sequences'
# (tên hiển thị, thư mục con trong ROOT, tổng số video)
# Tổng số DFD lấy từ misc/deepfake_detection_filenames.json trên server
TARGETS = [
    ('original c23', 'original_sequences/youtube/c23/videos', 1000),
    ('Deepfakes c23', FF + '/Deepfakes/c23/videos', 1000),
    ('Face2Face c23', FF + '/Face2Face/c23/videos', 1000),
    ('FaceSwap c23', FF + '/FaceSwap/c23/videos', 1000),
    ('NeuralTex c23', FF + '/NeuralTextures/c23/videos', 1000),
    ('Deepfakes mask', FF + '/Deepfakes/masks/videos', 1000),
    ('Face2Face mask', FF + '/Face2Face/masks/videos', 1000),
    ('FaceSwap mask', FF + '/FaceSwap/masks/videos', 1000),
    ('NeuralTex mask', FF + '/NeuralTextures/masks/videos', 1000),
    ('DFD mask', FF + '/DeepFakeDetection/masks/videos', 3068),
    ('DFD c23', FF + '/DeepFakeDetection/c23/videos', 3068),
    ('DFD original c23', 'original_sequences/actors/c23/videos', 363),
    ('original c40', 'original_sequences/youtube/c40/videos', 1000),
    ('Deepfakes c40', FF + '/Deepfakes/c40/videos', 1000),
    ('Face2Face c40', FF + '/Face2Face/c40/videos', 1000),
    ('FaceSwap c40', FF + '/FaceSwap/c40/videos', 1000),
    ('NeuralTex c40', FF + '/NeuralTextures/c40/videos', 1000),
    ('DFD c40', FF + '/DeepFakeDetection/c40/videos', 3068),
    ('DFD original c40', 'original_sequences/actors/c40/videos', 363),
]
BAR_WIDTH = 30
REFRESH_SECONDS = 3


def scan(folder):
    """Trả về (số file mp4, tổng dung lượng bytes) trong một thư mục."""
    count, size = 0, 0
    if os.path.isdir(folder):
        for entry in os.scandir(folder):
            if entry.name.endswith('.mp4'):
                count += 1
                size += entry.stat().st_size
    return count, size


def bar(done, total):
    filled = int(BAR_WIDTH * done / total)
    return '#' * filled + '-' * (BAR_WIDTH - filled)


def render(history):
    lines = ['FaceForensics++ - tien trinh tai', '']
    total_done, total_size = 0, 0
    for name, path, total in TARGETS:
        done, size = scan(os.path.join(ROOT, path))
        total_done += done
        total_size += size
        status = 'XONG' if done >= total else (
            'dang tai' if done > 0 else 'cho')
        lines.append('{:<17} [{}] {:>4}/{:<4} {:>5.1f}%  {}'.format(
            name, bar(done, total), done, total, 100 * done / total, status))

    grand_total = sum(t for _, _, t in TARGETS)
    lines.append('')
    lines.append('{:<17} [{}] {:>4}/{:<4} {:>5.1f}%  {:.2f} GB'.format(
        'TONG', bar(total_done, grand_total), total_done, grand_total,
        100 * total_done / grand_total, total_size / 1024 ** 3))

    # Ước lượng tốc độ và thời gian còn lại từ ~1 phút gần nhất
    now = time.time()
    history.append((now, total_done))
    while len(history) > 2 and now - history[0][0] > 60:
        history.pop(0)
    t0, d0 = history[0]
    if now - t0 > 0 and total_done > d0:
        rate = (total_done - d0) / (now - t0) * 60
        eta_min = (grand_total - total_done) / rate
        lines.append('Toc do: {:.1f} video/phut - con lai khoang {}h{:02d}m'
                     .format(rate, int(eta_min // 60), int(eta_min % 60)))
    else:
        lines.append('Toc do: dang do...')
    lines.append('')
    lines.append('Cap nhat: {}   (Ctrl+C de thoat)'.format(
        time.strftime('%H:%M:%S')))
    return '\n'.join(lines), total_done >= grand_total


def main():
    os.system('')  # bật mã ANSI trên Windows console
    history = []
    try:
        while True:
            text, finished = render(history)
            sys.stdout.write('\033[2J\033[H' + text + '\n')
            sys.stdout.flush()
            if finished:
                print('\nDa tai xong tat ca!')
                break
            time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
