"""Tiền xử lý video FF++/DFD của TUM: lấy frame, phát hiện mặt bằng RetinaFace, cắt ảnh và mask.

Chạy: python src/preprocess.py            (chạy lại sẽ bỏ qua video đã xử lý xong)
Thử nhanh: python src/preprocess.py --limit 5

Mỗi video: lấy `--frames` frame cách đều; đọc video và video mask song song từng frame
nên chỉ số frame luôn khớp. Mặt lớn nhất trong frame được nới khung `--scale` lần,
làm vuông, cắt ảnh và mask bằng đúng khung đó rồi resize về `--size`.
Mask nhị phân bằng ngưỡng > MASK_THRESHOLD (mask Face2Face/FaceSwap/NeuralTextures
là ảnh xám có bóng đổ, ngưỡng 127 làm mất 10 đến 20% vùng giả), lưu 0/255.

Kết quả: data/faces/<nhóm>/{frames,masks}/<video>/<chỉ số frame>.png
Khung cắt trên frame gốc: data/faces/<nhóm>/frames/<video>/boxes.json,
    {"<chỉ số frame>.png": [x, y, cạnh]} (khung vuông của crop_box, để ghép mask về frame)
Nhật ký từng video: data/faces/log.jsonl
Bản c40 (test độ bền, dùng chung mask), cắt đúng frame và khung của bản c23 để hai bản
chỉ khác nhau ở mức nén:
    python src/preprocess.py --compression c40 --out data/faces_c40 --boxes-from data/faces
"""
import argparse
import glob
import json
import multiprocessing as mp
import os

import cv2
import numpy as np
from tqdm import tqdm

MASK_THRESHOLD = 10
DETECT_MAX_SIDE = 960   # phát hiện mặt trên ảnh thu nhỏ, rồi đổi khung về ảnh gốc
SCORE_THRESHOLD = 0.9
# (nhóm, có mask không); video ở <nhóm>/<mức nén>/videos, mask ở <nhóm>/masks/videos
GROUPS = [
    ('original_sequences/youtube', False),
    ('original_sequences/actors', False),
] + [
    ('manipulated_sequences/' + m, True)
    for m in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures',
              'DeepFakeDetection']
]

_detector = None


def init_worker(gpu_id):
    global _detector
    cv2.setNumThreads(2)
    from batch_face import RetinaFace
    _detector = RetinaFace(gpu_id=gpu_id, network='resnet50')


def read_frames(video_path, mask_path, n_frames, wanted=None):
    """Đọc tuần tự video (và mask) cùng lúc, giữ các frame ở chỉ số cách đều
    (hoặc đúng các chỉ số trong `wanted`)."""
    cap = cv2.VideoCapture(video_path)
    mcap = cv2.VideoCapture(mask_path) if mask_path else None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    usable = total
    if mcap is not None:
        # 118 video DFD có mask ngắn hơn video (mask khớp từ frame đầu), nên chỉ
        # lấy frame trong đoạn có mask
        usable = min(total, int(mcap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if wanted is None:
        wanted = set(np.linspace(0, max(usable - 1, 0), n_frames).round().astype(int))
    frames, masks, idx = [], [], []
    for i in range(max(wanted) + 1):
        if not cap.grab() or (mcap is not None and not mcap.grab()):
            break
        if i in wanted:
            ok, f = cap.retrieve()
            ok_m, m = mcap.retrieve() if mcap is not None else (True, None)
            if ok and ok_m:
                frames.append(f)
                masks.append(m)
                idx.append(i)
    cap.release()
    if mcap is not None:
        mcap.release()
    return frames, masks, idx, total


def detect_faces(frames):
    """Mọi khung mặt (x1, y1, x2, y2) theo toạ độ ảnh gốc, cho từng frame."""
    small, scales = [], []
    for f in frames:
        s = min(1.0, DETECT_MAX_SIDE / max(f.shape[:2]))
        small.append(cv2.cvtColor(cv2.resize(f, None, fx=s, fy=s) if s < 1 else f,
                                  cv2.COLOR_BGR2RGB))
        scales.append(s)
    results = _detector(small, threshold=SCORE_THRESHOLD, cv=False, batch_size=8)
    return [[np.asarray(b, float) / s for b, _, _ in faces]
            for faces, s in zip(results, scales)]


def pick_face(boxes, binary_mask):
    """Video thật: mặt lớn nhất. Video giả: mặt chứa nhiều vùng mask nhất, vì cảnh
    DFD có thể có 2 người mà người lớn hơn không bị giả."""
    if not boxes:
        return None
    area = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    if binary_mask is None or not binary_mask.any():
        return boxes[int(np.argmax(area))]
    h, w = binary_mask.shape
    inside = []
    for b in boxes:
        x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
        x2, y2 = min(w, int(b[2]) + 1), min(h, int(b[3]) + 1)
        inside.append(int(binary_mask[y1:y2, x1:x2].sum()) if x2 > x1 and y2 > y1 else 0)
    return boxes[max(range(len(boxes)), key=lambda k: (inside[k], area[k]))]


def crop_box(box, scale):
    """Góc trái trên và cạnh của khung vuông = scale × cạnh dài của mặt."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * scale / 2
    return int(round(cx - half)), int(round(cy - half)), int(round(2 * half))


def square_crop(img, box, scale, size, interp):
    """Cắt khung `crop_box`; phần ra ngoài ảnh tô đen."""
    return cut_square(img, crop_box(box, scale), size, interp)


def cut_square(img, square, size, interp):
    """Cắt khung vuông (x, y, cạnh) rồi resize về size × size; phần ra ngoài ảnh tô đen."""
    l, t, side = square
    h, w = img.shape[:2]
    pad = max(0, -l, -t, l + side - w, t + side - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    crop = img[t + pad:t + pad + side, l + pad:l + pad + side]
    if interp is None:
        interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interp)


def process_video(task):
    group, video_path, mask_path, out_root, n_frames, scale, size, boxes_from = task
    video = os.path.splitext(os.path.basename(video_path))[0]
    if '%' in video:   # chuỗi ảnh (thư mục/%08d.png): tên video là tên thư mục
        video = os.path.basename(os.path.dirname(video_path))
    ref = None
    if boxes_from:   # dùng lại đúng frame và khung cắt của bản đã xử lý (c23)
        ref_path = os.path.join(boxes_from, group, 'frames', video, 'boxes.json')
        ref = {}
        if os.path.isfile(ref_path):
            with open(ref_path, encoding='utf-8') as fp:
                ref = json.load(fp)
        wanted = {int(name[:-4]) for name in ref}
        frames, masks, idx, total = read_frames(video_path, mask_path, n_frames, wanted) \
            if wanted else ([], [], [], 0)
        all_boxes = [None] * len(frames)
    else:
        frames, masks, idx, total = read_frames(video_path, mask_path, n_frames)
        all_boxes = detect_faces(frames) if frames else []
    frame_dir = os.path.join(out_root, group, 'frames', video)
    mask_dir = os.path.join(out_root, group, 'masks', video)
    os.makedirs(frame_dir, exist_ok=True)
    if mask_path:
        os.makedirs(mask_dir, exist_ok=True)
    saved = empty_mask = 0
    crops = {}
    for f, m, i, boxes in zip(frames, masks, idx, all_boxes):
        binary = (m.max(axis=2) > MASK_THRESHOLD).astype(np.uint8) \
            if mask_path else None
        name = '{:04d}.png'.format(i)
        if ref is not None:
            square = ref[name]
        else:
            box = pick_face(boxes, binary)
            if box is None:
                continue
            square = crop_box(box, scale)
        crops[name] = square
        cv2.imwrite(os.path.join(frame_dir, name),
                    cut_square(f, square, size, None),
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if mask_path:
            crop = cut_square(binary * 255, square, size, cv2.INTER_NEAREST)
            empty_mask += not crop.any()
            cv2.imwrite(os.path.join(mask_dir, name), crop)
        saved += 1
    with open(os.path.join(frame_dir, 'boxes.json'), 'w', encoding='utf-8') as fp:
        json.dump(crops, fp)
    return {'group': group, 'video': video, 'total_frames': total,
            'read': len(frames), 'saved': saved,
            'no_face': len(frames) - saved, 'empty_mask': empty_mask}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--src', default='data/FaceForensics')
    parser.add_argument('--out', default='data/faces')
    parser.add_argument('--compression', default='c23', choices=['c23', 'c40'])
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument('--scale', type=float, default=1.3)
    parser.add_argument('--size', type=int, default=224)   # Swin-T: 224 = 7 × 32
    parser.add_argument('--workers', type=int, default=3,
                        help='Số tiến trình, mỗi tiến trình một RetinaFace trên GPU.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Chỉ xử lý N video mỗi nhóm (để chạy thử).')
    parser.add_argument('--boxes-from', dest='boxes_from',
                        help='Thư mục đã xử lý (vd data/faces): lấy đúng frame và khung cắt '
                             'trong boxes.json của nó thay vì phát hiện mặt lại.')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, 'log.jsonl')

    done = set()
    if os.path.isfile(log_path):
        with open(log_path, encoding='utf-8') as f:
            done = {(r['group'], r['video']) for r in map(json.loads, f)}

    tasks = []
    for group, has_mask in GROUPS:
        videos = sorted(glob.glob(os.path.join(
            args.src, group, args.compression, 'videos', '*.mp4')))
        if args.limit:
            videos = videos[:args.limit]
        for v in videos:
            name = os.path.splitext(os.path.basename(v))[0]
            if (group, name) in done:
                continue
            mask = os.path.join(args.src, group, 'masks', 'videos',
                                name + '.mp4') if has_mask else None
            if has_mask and not os.path.isfile(mask):
                print('no mask video, skipped:', v)
                continue
            tasks.append((group, v, mask, args.out, args.frames, args.scale,
                          args.size, args.boxes_from))
    print('{} videos to process ({} already done)'.format(len(tasks), len(done)))

    # dùng khung có sẵn thì không cần nạp RetinaFace
    init = (None, ()) if args.boxes_from else (init_worker, (0,))
    with mp.Pool(args.workers, initializer=init[0], initargs=init[1]) as pool, \
            open(log_path, 'a', encoding='utf-8') as log:
        no_face = 0
        bar = tqdm(pool.imap_unordered(process_video, tasks), total=len(tasks))
        for r in bar:
            log.write(json.dumps(r) + '\n')
            log.flush()
            no_face += r['no_face']
            bar.set_postfix(no_face=no_face)


if __name__ == '__main__':
    main()
