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
Đoạn frame liên tiếp cho mô hình thời gian (4 đoạn × 8 frame, mỗi đoạn một khung cắt chung):
    python src/preprocess.py --out data/clips --clip-len 8
    python src/preprocess.py --compression c40 --out data/clips_c40 --boxes-from data/clips
Đoạn dài (2 đoạn × 16 frame, cách 1 frame lấy 1 ≈ 1,1–1,3 s: đủ một lần chớp mắt và một chu kỳ
miệng; bỏ frame không thấy mặt, khung cắt bám theo mặt khi người di chuyển):
    python src/preprocess.py --out data/clips16 --clip-len 16 --clip-stride 2 --follow-face
Các đoạn đặt ở giữa các phần bằng nhau của video (không sát đầu/cuối); --follow-face dịch khung
cắt vào trong video thay vì tô đen phần tràn mép.
DFD cắt lại cho mô hình từng khung (FF++ giữ data/faces): 32 khung đơn, chọn khung chính diện
gần các mốc cách đều, bỏ khung không thấy mặt:
    python src/preprocess.py --out data/faces_dfd --clip-len 1 --follow-face --frontal 0.3 \
        --groups actors DeepFakeDetection
    python src/preprocess.py --compression c40 --out data/faces_dfd_c40 --boxes-from data/faces_dfd \
        --groups actors DeepFakeDetection
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
# yaw_ratio ứng với |yaw| ≈ 30° (đo trên 4000 khung DFD so với MediaPipe: trung vị 0,30 ở
# 20–30°, 0,42 ở 30–45°; tương quan 0,84; ngưỡng 0,3 còn 2,4% khung |yaw| ≥ 30°)
FRONTAL_RATIO = 0.3
MOVE_LIMIT = 0.15      # --follow-face: mặt dịch quá 15% cạnh khung trong đoạn → khung riêng từng frame
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


def centered(last, n):
    """n mốc ở giữa n phần bằng nhau của [0, last]: không rơi vào sát đầu/cuối video, nơi người
    trong DFD đang bước vào/ra khung hình (mặt tràn mép, bị che)."""
    return (np.arange(n) + 0.5) * last / n


def inside(square, w, h):
    """Dịch khung vuông (x, y, cạnh) vào trong ảnh w×h, giữ cạnh: mặt sát mép thì lệch tâm một
    chút thay vì có dải đen (phần ngoài video). Khung to hơn ảnh thì giữ nguyên."""
    x, y, s = square
    if s > min(w, h):
        return square
    return min(max(x, 0), w - s), min(max(y, 0), h - s), s


def read_frames(video_path, mask_path, n_frames, wanted=None, clip_len=0, stride=1):
    """Đọc tuần tự video (và mask) cùng lúc, giữ các frame ở chỉ số cách đều
    (hoặc đúng các chỉ số trong `wanted`). `clip_len` > 0: n_frames // clip_len đoạn,
    mỗi đoạn clip_len frame cách nhau `stride` frame, các đoạn cách đều trong video."""
    cap = cv2.VideoCapture(video_path)
    mcap = cv2.VideoCapture(mask_path) if mask_path else None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    usable = total
    if mcap is not None:
        # 118 video DFD có mask ngắn hơn video (mask khớp từ frame đầu), nên chỉ
        # lấy frame trong đoạn có mask
        usable = min(total, int(mcap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if wanted is None and clip_len:
        span = (clip_len - 1) * stride + 1
        starts = centered(max(usable - span, 0), n_frames // clip_len).round().astype(int)
        wanted = {s + k * stride for s in starts for k in range(clip_len) if s + k * stride < usable}
    elif wanted is None:
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


def detect_faces(frames, with_kps=False):
    """Mọi khung mặt (x1, y1, x2, y2) theo toạ độ ảnh gốc, cho từng frame.
    with_kps: mỗi mặt là (khung, 5 điểm mốc mắt trái, mắt phải, mũi, hai khoé miệng)."""
    small, scales = [], []
    for f in frames:
        s = min(1.0, DETECT_MAX_SIDE / max(f.shape[:2]))
        small.append(cv2.cvtColor(cv2.resize(f, None, fx=s, fy=s) if s < 1 else f,
                                  cv2.COLOR_BGR2RGB))
        scales.append(s)
    results = _detector(small, threshold=SCORE_THRESHOLD, cv=False, batch_size=8)
    if with_kps:
        return [[(np.asarray(b, float) / s, np.asarray(p, float) / s) for b, p, _ in faces]
                for faces, s in zip(results, scales)]
    return [[np.asarray(b, float) / s for b, _, _ in faces]
            for faces, s in zip(results, scales)]


def yaw_ratio(kps):
    """Độ lệch của mũi khỏi trung điểm hai mắt, chia khoảng cách hai mắt: 0 khi nhìn thẳng,
    tăng theo |góc quay|. Ngưỡng tương ứng |yaw| MediaPipe đo trong FRONTAL_RATIO."""
    (lx, _), (rx, _), (nx, _) = kps[0], kps[1], kps[2]
    return abs(nx - (lx + rx) / 2) / max(abs(rx - lx), 1.0)


def frontal_starts(video_path, usable, n_clips, clip_len, max_ratio, step=1, stride=6):
    """Điểm bắt đầu của n_clips đoạn mặt chính diện: đọc lướt mỗi `stride` frame, đo yaw_ratio
    của mặt lớn nhất; điểm số của một đoạn = yaw_ratio lớn nhất trong các frame lấy mẫu của nó
    (không thấy mặt = vô cực). Mỗi đoạn chọn gần vị trí cách đều nhất trong các đoạn đạt
    ngưỡng; hết đoạn đạt ngưỡng thì lấy đoạn ít nghiêng nhất còn lại (video nào cũng đủ đoạn).
    Đoạn gồm clip_len frame cách nhau `step`; các đoạn cách nhau ≥ một đoạn để temporal.py
    không gộp nhầm hai đoạn.
    shortcut: video giả 2 người đo góc mặt lớn nhất, không theo mask; đổi nếu thấy chọn nhầm."""
    cap = cv2.VideoCapture(video_path)
    ratio, batch = {}, []

    def flush():
        for (i, _), faces in zip(batch, detect_faces([f for _, f in batch], with_kps=True)):
            if faces:
                b, p = max(faces, key=lambda t: (t[0][2] - t[0][0]) * (t[0][3] - t[0][1]))
                ratio[i] = yaw_ratio(p)
        batch.clear()

    for i in range(usable):
        if not cap.grab():
            break
        if i % stride == 0:
            ok, f = cap.retrieve()
            if ok:
                batch.append((i, f))
            if len(batch) == 16:
                flush()
    flush()
    cap.release()
    span = (clip_len - 1) * step + 1
    last = max(usable - span, 0)
    score = {s: max((ratio.get(i, np.inf) for i in range(s, s + span) if i % stride == 0),
                    default=np.inf) for s in range(last + 1)}
    chosen = []
    for t in centered(last, n_clips):
        free = [s for s in score if all(abs(s - c) >= 2 * span for c in chosen)]
        if not free:
            break
        ok = [s for s in free if score[s] < max_ratio]
        chosen.append(min(ok, key=lambda s: abs(s - t)) if ok
                      else min(free, key=lambda s: (score[s], abs(s - t))))
    return {s + k * step for s in chosen for k in range(clip_len) if s + k * step < usable}


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


def clip_boxes(idx, all_boxes, masks, mask_path, follow=False, stride=1):
    """Một khung mặt chung (trung vị) cho mỗi đoạn frame liên tiếp: khung phát hiện riêng từng
    frame lệch vài pixel, cắt theo nó tạo ra rung giả mà mô hình thời gian sẽ học nhầm.
    follow: frame không thấy mặt bị bỏ (không cắt nền theo khung chung), và đoạn có mặt dịch
    > MOVE_LIMIT cạnh khung so với trung vị thì mỗi frame dùng khung riêng, làm trơn 3 frame,
    để mặt không trôi ra mép (người đi lại trong DFD).
    Trả về {chỉ số frame: khung hoặc None (không cắt)}."""
    out, clip = {}, []
    for k, i in enumerate(idx + [None]):
        if clip and (i is None or i != clip[-1][0] + stride):    # hết một đoạn
            picked = [b for _, b in clip if b is not None]
            box = np.median(picked, axis=0) if picked else None
            out.update({j: box for j, _ in clip})
            if follow and picked:
                side = max(box[2] - box[0], box[3] - box[1])
                moving = max(np.abs((b[:2] + b[2:]) / 2 - (box[:2] + box[2:]) / 2).max()
                             for b in picked) > MOVE_LIMIT * side
                for n, (j, b) in enumerate(clip):
                    near = [c for _, c in clip[max(0, n - 1):n + 2] if c is not None]
                    out[j] = None if b is None else np.mean(near, 0) if moving else box
            clip = []
        if i is not None:
            binary = (masks[k].max(axis=2) > MASK_THRESHOLD) if mask_path else None
            clip.append((i, pick_face(all_boxes[k], binary)))
    return out


def process_video(task):
    (group, video_path, mask_path, out_root, n_frames, scale, size, boxes_from, clip_len,
     follow, frontal, stride) = task
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
        wanted = None
        if frontal:
            count = lambda p: int(cv2.VideoCapture(p).get(cv2.CAP_PROP_FRAME_COUNT))  # noqa: E731
            usable = min(count(video_path), count(mask_path)) if mask_path else count(video_path)
            wanted = frontal_starts(video_path, usable, n_frames // clip_len, clip_len, frontal,
                                    stride)
        frames, masks, idx, total = read_frames(video_path, mask_path, n_frames, wanted,
                                                clip_len=clip_len, stride=stride)
        all_boxes = detect_faces(frames) if frames else []
    shared = clip_boxes(idx, all_boxes, masks, mask_path, follow, stride) \
        if clip_len and ref is None else None
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
        elif shared is not None:
            if shared.get(i) is None:
                continue
            square = crop_box(shared[i], scale)
            if follow:
                square = inside(square, f.shape[1], f.shape[0])
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
    parser.add_argument('--clip-len', dest='clip_len', type=int, default=0,
                        help='> 0: lấy frames/clip_len đoạn frame liên tiếp (cho mô hình thời '
                             'gian), mỗi đoạn một khung cắt chung.')
    parser.add_argument('--clip-stride', dest='clip_stride', type=int, default=1,
                        help='với --clip-len: lấy 1 frame mỗi N frame trong đoạn (đoạn dài hơn '
                             'mà không tăng số frame), vd --clip-len 16 --clip-stride 2')
    parser.add_argument('--follow-face', dest='follow', action='store_true',
                        help='với --clip-len: bỏ frame không thấy mặt; đoạn mặt di chuyển nhiều '
                             'dùng khung cắt riêng từng frame (xem clip_boxes).')
    parser.add_argument('--frontal', type=float, default=0,
                        help='> 0 (với --clip-len): chọn đoạn ở chỗ mặt chính diện, ngưỡng '
                             'yaw_ratio (xem FRONTAL_RATIO), thay cho các đoạn cách đều.')
    parser.add_argument('--groups', nargs='+',
                        help='chỉ xử lý các nhóm chứa chuỗi này, vd: actors DeepFakeDetection')
    args = parser.parse_args()
    assert not args.frontal or args.clip_len, '--frontal cần --clip-len'
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, 'log.jsonl')

    done = set()
    if os.path.isfile(log_path):
        with open(log_path, encoding='utf-8') as f:
            done = {(r['group'], r['video']) for r in map(json.loads, f)}

    tasks = []
    for group, has_mask in GROUPS:
        if args.groups and not any(g in group for g in args.groups):
            continue
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
                          args.size, args.boxes_from, args.clip_len, args.follow, args.frontal,
                          args.clip_stride))
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
