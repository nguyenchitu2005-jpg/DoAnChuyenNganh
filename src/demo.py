"""Demo Gradio: tải video lên; Mô hình 1 tô vùng mặt bị chỉnh sửa trên từng frame,
Mô hình 2 (train_ai.py, nếu đã train) cho biết video có do AI tạo không, kèm Grad-CAM.

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
from matplotlib import colormaps
import torch
from scipy.ndimage import uniform_filter1d

import preprocess
from dataset import eval_transform
import train_ai
from evaluate import load_checkpoint
from visualize import PROB_CMAP

MAX_FRAMES = 300       # video dài hơn thì lấy cách đều 300 frame
SCALE, SIZE = 1.3, 256  # giống lúc tiền xử lý dữ liệu train
SMOOTH = 5            # số frame lấy trung bình xác suất
FA_AREA = 0.01        # giống metrics.py: frame bị tô > 1% diện tích mặt là có vùng giả
COLOR = (36, 36, 220)  # BGR, đỏ: phủ bán trong suốt (alpha 0,5) lên vùng dự đoán
AI_RUN = train_ai.RUN_DIR
# mô hình segmentation (Mô hình 1); Mô hình 2 ở AI_RUN là bộ phân loại, không chọn ở đây
CHECKPOINTS = {os.path.basename(os.path.dirname(p)): p
               for p in sorted(glob.glob('runs/*/best.pt'))
               if 'smoke' not in p and os.path.dirname(p).replace(os.sep, '/') != AI_RUN}
_models = {}


def get_ai_model(device):
    """Mô hình 2 và quy tắc kết luận, hoặc (None, None) nếu chưa train."""
    rule_path = os.path.join(AI_RUN, 'video_rule.json')
    if not os.path.exists(rule_path):
        return None, None
    if 'ai' not in _models:
        model = train_ai.build_model(pretrained=False)
        model.load_state_dict(torch.load(os.path.join(AI_RUN, 'best.pt'),
                                         map_location='cpu')['model'])
        with open(rule_path, encoding='utf-8') as f:
            _models['ai'] = (model.to(device).eval(), json.load(f))
    return _models['ai']


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
def predict(crops, model, device, tf=None):
    tf = tf or eval_transform(SIZE)
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
    ai_model, ai_rule = get_ai_model(device)
    # bộ phân loại ra (B, 1) nên predict trả về một xác suất mỗi frame
    ai_probs = (predict(crops, ai_model, device, train_ai.ai_transform(False))
                if ai_model and crops else None)
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
            # viền đậm để vùng phủ đỏ vẫn rõ trên màu da
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
    ])
    if ai_probs is not None:
        ai = np.full(len(frames), np.nan)
        ai[found] = ai_probs
        df = pd.concat([df, pd.DataFrame({'Thời gian (giây)': t, 'Giá trị (%)': 100 * ai,
                                          'Chỉ số': 'Điểm do AI tạo (Mô hình 2)'})])
    df = df.dropna()
    gallery = []
    for k in pick_frames(probs):
        caption = 'giây {:.1f}: xác suất giả lớn nhất {:.2f}, vùng giả {:.0%}'.format(
            t[found[k]], peak[found[k]], area[found[k]])
        cam = None
        if ai_probs is not None:
            x = train_ai.ai_transform(False)(
                image=cv2.cvtColor(crops[k], cv2.COLOR_BGR2RGB))
            cam, _ = train_ai.gradcam(ai_model, x['image'][None].to(device))
            caption += ', điểm AI {:.2f}'.format(ai_probs[k])
        gallery.append((panel(crops[k], probs[k], thr, cam), caption))
    summary = verdict(model_name, len(frames), area[found], peak[found], t[found])
    if ai_probs is not None:
        summary += '\n\n---\n\n' + ai_verdict(ai_probs, ai_rule)
    return out_path, df, gallery, summary


def ai_verdict(ai_probs, rule):
    """Mô hình 2: video có do AI tạo không, theo điểm trung bình các frame."""
    score = float(np.mean(ai_probs))
    label = ('🟣 CÓ DẤU HIỆU DO AI TẠO' if score >= rule['threshold']
             else '⚪ KHÔNG ĐỦ DẤU HIỆU DO AI TẠO')
    test, extra = rule['test'], rule['extra']
    seen = ', '.join('{} {:.0%}'.format(k, v) for k, v in test['correct_by_source'].items()
                     if k != 'real_ff++')
    unseen = ', '.join('{} {:.0%}'.format(k, v)
                       for k, v in extra['correct_by_source'].items()
                       if not k.startswith('real'))
    return '\n'.join([
        '## Mô hình 2: ' + label,
        '**Bằng chứng**',
        '- Điểm do AI tạo trung bình **{:.2f}** (ngưỡng {:.2f}, chọn trên tập val '
        'TalkingHeadBench); **{:.0%}** số frame vượt ngưỡng.'.format(
            score, rule['threshold'], np.mean(ai_probs >= rule['threshold'])),
        '- Ô cuối mỗi ảnh bên dưới là Grad-CAM: **vùng mô hình dựa vào để quyết định** '
        '(vàng = nhiều), không phải vùng bị làm giả.',
        '', '**Độ tin cậy (EfficientNet-B0, ảnh mặt)**', '',
        '| Tập kiểm tra | Nhận đúng |', '| --- | --- |',
        '| Video thật FF++ test | {:.1%} |'.format(test['real_correct']),
        '| Video AI, bộ sinh đã gặp, người mới | {:.1%} ({}) |'.format(
            test['ai_correct'], seen),
        '| Video AI, bộ sinh chưa gặp | {:.1%} ({}) |'.format(extra['ai_correct'], unseen),
        '| Video thật khác nguồn (diễn viên DFD) | {:.1%} |'.format(extra['real_correct']),
        '', 'Mô hình 2 học từ video talking-head (ảnh chân dung được AI làm cử động). Video '
            'AI loại khác (Sora, filter AI...) có thể bị bỏ sót.',
    ])


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


def panel(crop, prob, thr, cam=None):
    """Ảnh mặt | bản đồ xác suất từng pixel (thang cố định 0-1) | vùng vượt ngưỡng
    | Grad-CAM của Mô hình 2 phủ lên ảnh (nếu có)."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    heat = (PROB_CMAP(prob)[..., :3] * 255).astype(np.uint8)
    # viền xám để ô toàn 0 (mặt thật) không trông như ảnh trống
    cv2.rectangle(heat, (0, 0), (SIZE - 1, SIZE - 1), (195, 194, 183), 2)
    over = rgb.copy()
    m = prob > thr
    over[m] = (0.5 * over[m] + 0.5 * np.array(COLOR[::-1])).astype(np.uint8)
    gap = np.full((SIZE, 6, 3), 255, np.uint8)
    tiles = [rgb, gap, heat, gap, over]
    if cam is not None:
        # thang màu khác hẳn Mô hình 1 để không đọc nhầm thành "vùng giả"
        color = (colormaps['plasma'](cam)[..., :3] * 255).astype(np.uint8)
        # Mô hình 2 chỉ nhìn khung mặt (cắt giữa), nên phủ lên đúng phần đó
        c = (SIZE - train_ai.FACE) // 2
        face = cv2.resize(rgb[c:SIZE - c, c:SIZE - c], (SIZE, SIZE))
        tiles += [gap, (0.5 * face + 0.5 * color).astype(np.uint8)]
    return np.hstack(tiles)


with gr.Blocks(title='Phát hiện vùng mặt bị chỉnh sửa') as app:
    gr.Markdown('# Phát hiện vùng mặt bị chỉnh sửa và video do AI tạo\n'
                'Tải video lên. Mô hình 1: vùng phủ đỏ là vùng bị ghép/sửa. '
                'Mô hình 2: video có do AI tạo không, kèm Grad-CAM.')
    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label='Video đầu vào')
            model_in = gr.Dropdown(list(CHECKPOINTS), label='Mô hình',
                                   value='unet_effb0' if 'unet_effb0' in CHECKPOINTS
                                   else next(iter(CHECKPOINTS), None),
                                   info='Mô hình 1 (segmentation)')
            button = gr.Button('Phân tích', variant='primary')
        with gr.Column():
            video_out = gr.Video(label='Kết quả')
            summary_out = gr.Markdown()
    gallery_out = gr.Gallery(
        label='Ảnh mặt | Mô hình 1: xác suất giả từng pixel (trắng = 0, xanh đậm = 1) | '
              'vùng vượt ngưỡng 0,5 (đỏ) | Mô hình 2: Grad-CAM (vàng = mô hình dựa vào '
              'nhiều)', columns=2, height='auto',
        object_fit='contain')   # mặc định 'cover' cắt ảnh 3 ô, chỉ còn ô giữa
    plot_out = gr.LinePlot(x='Thời gian (giây)', y='Giá trị (%)', color='Chỉ số',
                           label='Vùng giả và xác suất giả theo thời gian',
                           y_lim=[0, 100])
    button.click(run, [video_in, model_in],
                 [video_out, plot_out, gallery_out, summary_out])

if __name__ == '__main__':
    app.launch()
