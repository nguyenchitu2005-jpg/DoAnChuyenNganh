"""Demo Gradio: tải video lên, mô hình tô vùng mặt bị chỉnh sửa trên từng frame.

Chạy: python src/demo.py   rồi mở http://127.0.0.1:7860
Mỗi frame được cắt mặt giống hệt lúc train (RetinaFace, nới khung 1,3 lần, 256×256,
xem preprocess.py), dự đoán mask, rồi dán ngược mask lên frame gốc.
Kết quả: kết luận thật/giả kèm bằng chứng (quy tắc và ngưỡng trong <run>/video_rule.json,
tạo bằng `analyze.py --split video`), video có tô vùng giả, bản đồ xác suất, biểu đồ.
"""
import glob
import json
import os
import tempfile

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter1d

import preprocess
from dataset import eval_transform
from evaluate import load_checkpoint
from visualize import PROB_CMAP

MAX_FRAMES = 300       # video dài hơn thì lấy cách đều 300 frame
SCALE, SIZE = 1.3, 256  # giống lúc tiền xử lý dữ liệu train
SMOOTH = 5            # số frame lấy trung bình xác suất
FA_AREA = 0.01        # giống metrics.py: frame bị tô > 1% diện tích mặt là có vùng giả
COLOR = (52, 104, 235)  # BGR của #eb6834 (cam, màu "tô thừa" trong visualize.py)
CHECKPOINTS = {os.path.basename(os.path.dirname(p)): p
               for p in sorted(glob.glob('runs/*/best.pt')) if 'smoke' not in p}
_models = {}


def get_model(name):
    if name not in _models:
        _models[name] = load_checkpoint(CHECKPOINTS[name])
    if preprocess._detector is None:
        preprocess.init_worker(0)
    return _models[name]


def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, -(-total // MAX_FRAMES))   # làm tròn lên
    frames, i = [], 0
    while len(frames) < MAX_FRAMES and cap.grab():
        if i % step == 0:
            frames.append(cap.retrieve()[1])
        i += 1
    cap.release()
    return frames, fps / step


def paste_back(prob, box, shape):
    """Ngược với square_crop: phóng xác suất 256×256 về khung mặt trên frame gốc."""
    l, t, side = preprocess.crop_box(box, SCALE)
    h, w = shape[:2]
    full = np.zeros((h, w), np.float32)
    p = cv2.resize(prob, (side, side))
    x1, y1, x2, y2 = max(l, 0), max(t, 0), min(l + side, w), min(t + side, h)
    if x2 > x1 and y2 > y1:
        full[y1:y2, x1:x2] = p[y1 - t:y2 - t, x1 - l:x2 - l]
    return full


@torch.no_grad()
def predict(crops, model, device):
    tf = eval_transform(SIZE)
    probs = []
    for k in range(0, len(crops), 32):
        x = torch.stack([tf(image=cv2.cvtColor(c, cv2.COLOR_BGR2RGB))['image']
                         for c in crops[k:k + 32]]).to(device)
        with torch.autocast(device.type, dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            probs.append(torch.sigmoid(model(x).float())[:, 0].cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros((0, SIZE, SIZE))


def run(video, model_name, progress=gr.Progress()):
    if not video:
        raise gr.Error('Hãy tải lên một video.')
    model, cfg, _, device = get_model(model_name)
    progress(0, desc='Đọc video')
    frames, fps = read_video(video)
    if not frames:
        raise gr.Error('Không đọc được video.')
    progress(0.2, desc='Phát hiện mặt (RetinaFace)')
    boxes = []
    for k in range(0, len(frames), 32):
        # video thường chỉ có một người: lấy mặt lớn nhất như với video thật lúc train
        boxes += [preprocess.pick_face(b, None)
                  for b in preprocess.detect_faces(frames[k:k + 32])]
    found = [i for i, b in enumerate(boxes) if b is not None]
    progress(0.5, desc='Dự đoán mask')
    crops = [preprocess.square_crop(frames[i], boxes[i], SCALE, SIZE, None)
             for i in found]
    probs = predict(crops, model, device)
    # trung bình xác suất 5 frame liền kề (ảnh mặt đã căn theo khung mặt nên gần khớp
    # nhau), giảm nhấp nháy và ảnh hưởng của frame khó như frame đầu/cuối video
    if len(probs):
        probs = uniform_filter1d(probs, SMOOTH, axis=0, mode='nearest')

    thr = cfg['threshold']
    area = np.full(len(frames), np.nan)
    out_path = os.path.join(tempfile.mkdtemp(), 'ket_qua.webm')
    h, w = frames[0].shape[:2]
    # OpenCV pip không ghi được H.264; VP8/WebM thì trình duyệt phát được
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'VP80'), fps, (w, h))
    probs_by_frame = dict(zip(found, probs))
    for i, f in enumerate(progress.tqdm(frames, desc='Ghi video')):
        f = f.copy()
        if i in probs_by_frame:
            prob = probs_by_frame[i]
            area[i] = (prob > thr).mean()
            mask = paste_back(prob, boxes[i], f.shape) > thr
            f[mask] = (0.5 * f[mask] + 0.5 * np.array(COLOR)).astype(np.uint8)
            # viền đậm vì lớp tô cam khó thấy trên màu da
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(f, contours, -1, COLOR, 3)
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(f, (x1, y1), (x2, y2), (255, 255, 255), 2)
            text = 'vung gia {:.0%}'.format(area[i])
        else:
            text = 'khong thay mat'
        cv2.putText(f, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(f)
    writer.release()

    peak = np.full(len(frames), np.nan)
    peak[found] = probs.reshape(len(found), -1).max(1) if found else []
    t = np.arange(len(frames)) / fps
    df = pd.concat([
        pd.DataFrame({'Thời gian (giây)': t, 'Giá trị (%)': 100 * area,
                      'Chỉ số': 'Diện tích vùng giả (% mặt)'}),
        pd.DataFrame({'Thời gian (giây)': t, 'Giá trị (%)': 100 * peak,
                      'Chỉ số': 'Xác suất giả lớn nhất trên mặt'}),
    ]).dropna()
    gallery = [(panel(crops[k], probs[k], thr), 'giây {:.1f}: xác suất lớn nhất {:.2f}, '
                'vùng giả {:.0%}'.format(t[found[k]], peak[found[k]], area[found[k]]))
               for k in pick_frames(probs)]
    summary = verdict(model_name, len(frames), area[found], peak[found], t[found])
    return out_path, df, gallery, summary


def verdict(model_name, n_frames, area, peak, t):
    """Kết luận thật/giả cả video kèm bằng chứng và độ tin cậy đo trên test, DFD."""
    head = '**{}** frame đã xử lý, {} frame không thấy mặt.\n\n'.format(
        n_frames, n_frames - len(area))
    if not len(area):
        return head + '## Không kết luận được: không tìm thấy khuôn mặt.'
    flagged = (area > FA_AREA).mean()
    path = os.path.join(os.path.dirname(CHECKPOINTS[model_name]), 'video_rule.json')
    if not os.path.exists(path):
        return head + ('Chưa có quy tắc kết luận cho mô hình này: chạy '
                       '`python src/analyze.py {} --split video`.'.format(
                           CHECKPOINTS[model_name]))
    with open(path, encoding='utf-8') as f:
        rule = json.load(f)
    # quy tắc chọn trên 32 frame/video chưa làm trơn; demo dùng tới 300 frame đã
    # làm trơn nên tỉ lệ frame có thể lệch chút ít so với lúc chọn ngưỡng
    # dưới ngưỡng không chứng minh được video là thật, nên không ghi "THẬT"
    if flagged >= rule['threshold']:
        label = '🟠 GIẢ (có vùng mặt bị chỉnh sửa)'
    elif flagged >= rule['real_p95']:
        label = ('🟡 NGHI NGỜ (nhiều frame bị tô hơn 95% video thật, '
                 'nhưng chưa tới ngưỡng kết luận giả)')
    else:
        label = '⚪ KHÔNG ĐỦ DẤU HIỆU GIẢ'
    k = int(np.nanargmax(peak))
    lines = [
        head + '## Kết luận: ' + label,
        '**Bằng chứng**',
        '- **{:.0%}** số frame ({}/{}) có vùng giả (> 1% diện tích mặt). Ngưỡng kết luận '
        'giả: ≥ **{:.0%}** số frame, chọn trên tập FF++ val; 95% video thật của val '
        'có tỉ lệ dưới {:.0%}.'.format(flagged, int((area > FA_AREA).sum()), len(area),
                                       rule['threshold'], rule['real_p95']),
        '- Diện tích vùng giả trung bình **{:.1%}** mặt; xác suất giả cao nhất **{:.2f}** '
        '(giây {:.1f}).'.format(np.mean(area), peak[k], t[k]),
        '- Vị trí từng pixel bị cho là giả: xem bản đồ xác suất bên dưới và video kết quả.',
    ]
    if abs(flagged - rule['threshold']) < 0.1:
        lines.append('- ⚠️ Tỉ lệ frame gần ngưỡng: kết luận kém chắc chắn.')
    lines += [
        '', '**Độ tin cậy của quy tắc này ({})**'.format(model_name), '',
        '| Tập kiểm tra | Nhận đúng video thật | Nhận đúng video giả |',
        '| --- | --- | --- |',
        '| FF++ test (cùng nguồn với dữ liệu train) | {:.1%} | {:.1%} |'.format(
            rule['test']['real_correct'], rule['test']['fake_correct']),
        '| DFD (deepfake của Google, nguồn khác) | {:.1%} | {:.1%} |'.format(
            rule['dfd']['real_correct'], rule['dfd']['fake_correct']),
        '', 'Mô hình chỉ học từ FF++ (video tin tức, 4 phương pháp ghép/sửa mặt). Video '
            'do AI tạo trọn vẹn, filter làm đẹp, video quay bằng điện thoại/webcam hoặc '
            'deepfake tạo bằng công cụ khác có thể bị bỏ sót hoặc kết luận sai.',
    ]
    return '\n'.join(lines)


def pick_frames(probs, n=6):
    """n frame cách đều, thay frame cuối bằng frame có xác suất cao nhất."""
    if not len(probs):
        return []
    ks = list(np.linspace(0, len(probs) - 1, n - 1).round().astype(int))
    ks.append(int(probs.reshape(len(probs), -1).max(1).argmax()))
    return sorted(set(ks))


def panel(crop, prob, thr):
    """Ảnh mặt | bản đồ xác suất từng pixel (thang cố định 0-1) | vùng vượt ngưỡng."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    heat = (PROB_CMAP(prob)[..., :3] * 255).astype(np.uint8)
    # viền xám để ô toàn 0 (mặt thật) không trông như ảnh trống
    cv2.rectangle(heat, (0, 0), (SIZE - 1, SIZE - 1), (195, 194, 183), 2)
    over = rgb.copy()
    m = prob > thr
    over[m] = (0.5 * over[m] + 0.5 * np.array(COLOR[::-1])).astype(np.uint8)
    gap = np.full((SIZE, 6, 3), 255, np.uint8)
    return np.hstack([rgb, gap, heat, gap, over])


with gr.Blocks(title='Phát hiện vùng mặt bị chỉnh sửa') as app:
    gr.Markdown('# Phát hiện vùng mặt bị chỉnh sửa (deepfake segmentation)\n'
                'Tải video lên; vùng tô cam là vùng mô hình cho là bị chỉnh sửa.')
    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label='Video đầu vào')
            model_in = gr.Dropdown(list(CHECKPOINTS), label='Mô hình',
                                   value='unet_effb0' if 'unet_effb0' in CHECKPOINTS
                                   else next(iter(CHECKPOINTS), None))
            button = gr.Button('Phân tích', variant='primary')
        with gr.Column():
            video_out = gr.Video(label='Kết quả')
            summary_out = gr.Markdown()
    gallery_out = gr.Gallery(
        label='Từng pixel: ảnh mặt | xác suất giả (trắng = 0, xanh đậm = 1) | '
              'vùng vượt ngưỡng 0,5 (cam)', columns=2, height='auto',
        object_fit='contain')   # mặc định 'cover' cắt ảnh 3 ô, chỉ còn ô giữa
    plot_out = gr.LinePlot(x='Thời gian (giây)', y='Giá trị (%)', color='Chỉ số',
                           label='Vùng giả và xác suất giả theo thời gian',
                           y_lim=[0, 100])
    button.click(run, [video_in, model_in],
                 [video_out, plot_out, gallery_out, summary_out])

if __name__ == '__main__':
    app.launch()
