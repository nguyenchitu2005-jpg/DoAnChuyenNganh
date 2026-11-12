"""Demo Gradio: tải video lên; mô hình kết luận thật/giả và tô vùng mặt bị chỉnh sửa.

Chạy: python src/demo.py   rồi mở http://127.0.0.1:7860
Mỗi frame được cắt mặt giống hệt lúc train (RetinaFace, nới khung 1,3 lần, xem
preprocess.py). Kết luận cả video = trung bình xác suất giả của đầu phân loại trên các
frame có mặt, ngưỡng 0,5: đúng cách tính mức video của evaluate.py, nên bảng độ tin cậy
(đọc từ runs/<mô hình>/eval_*.json) khớp với cách demo kết luận.
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
import yaml
from scipy.ndimage import uniform_filter1d

import preprocess
from dataset import eval_transform
from evaluate import load_checkpoint
from visualize import PROB_CMAP

MAX_FRAMES = 300       # video dài hơn thì lấy cách đều 300 frame
SCALE = 1.3            # giống lúc tiền xử lý dữ liệu train
SMOOTH = 5             # số frame lấy trung bình xác suất
COLOR = (36, 36, 220)  # BGR, đỏ: phủ bán trong suốt (alpha 0,5) lên vùng dự đoán
CLASS_NAMES = ['Thật', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
DEFAULT_RUN = 'b1_mix'   # phân loại tốt nhất trên test (results/cls.json)
FF = 'data/FaceForensics'
EXAMPLES = [p for p in [f'{FF}/original_sequences/youtube/c23/videos/953.mp4'] + [
    f'{FF}/manipulated_sequences/{m}/c23/videos/953_974.mp4'
    for m in CLASS_NAMES[1:]] if os.path.isfile(p)]   # video thuộc tập test, mô hình chưa thấy


def list_models():
    """Chỉ các run có đầu phân loại (multitask), nhãn lấy từ results/models.json."""
    try:
        with open('results/models.json', encoding='utf-8') as f:
            labels = {r['model']: r['label'] for r in json.load(f)}
    except OSError:
        labels = {}
    out = {}
    for p in sorted(glob.glob('runs/*/best.pt')):
        run = os.path.basename(os.path.dirname(p))
        cfg_path = os.path.join(os.path.dirname(p), 'config.yaml')
        if 'smoke' in run or not os.path.isfile(cfg_path):
            continue
        with open(cfg_path, encoding='utf-8') as f:
            if not yaml.safe_load(f).get('multitask'):
                continue
        if run not in labels:   # chưa có trong bảng kết quả: thường là run đang train dở
            with open(os.path.join(os.path.dirname(p), 'history.jsonl'), encoding='utf-8') as f:
                labels[run] = f'{run} (đang train, đã {sum(1 for _ in f)} epoch)'
        out[f'{labels[run]}  [{run}]'] = run
    # mô hình mặc định đứng đầu danh sách
    return dict(sorted(out.items(), key=lambda kv: kv[1] != DEFAULT_RUN))


MODELS = list_models()
_models = {}


def get_model(run):
    if run not in _models:
        _models[run] = load_checkpoint(f'runs/{run}/best.pt')
    if preprocess._detector is None:
        preprocess.init_worker(0)
    return _models[run]


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
    """Ngược với square_crop: phóng xác suất của ảnh mặt về khung mặt trên frame gốc."""
    l, t, side = preprocess.crop_box(box, SCALE)
    h, w = shape[:2]
    full = np.zeros((h, w), np.float32)
    p = cv2.resize(prob, (side, side))
    x1, y1, x2, y2 = max(l, 0), max(t, 0), min(l + side, w), min(t + side, h)
    if x2 > x1 and y2 > y1:
        full[y1:y2, x1:x2] = p[y1 - t:y2 - t, x1 - l:x2 - l]
    return full


@torch.no_grad()
def predict(crops, model, device, size):
    """Mask xác suất (N,H,W), xác suất giả (N,) và xác suất 5 lớp (N,5) cho từng ảnh mặt."""
    tf = eval_transform(size)
    masks, fake, cls = [], [], []
    for k in range(0, len(crops), 32):
        x = torch.stack([tf(image=cv2.cvtColor(c, cv2.COLOR_BGR2RGB))['image']
                         for c in crops[k:k + 32]]).to(device)
        with torch.autocast(device.type, dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            seg, b, c = model(x, with_cls=True)
        masks.append(torch.sigmoid(seg.float())[:, 0].cpu().numpy())
        fake.append(torch.sigmoid(b.float()).cpu().numpy())
        cls.append(torch.softmax(c.float(), 1).cpu().numpy())
    return np.concatenate(masks), np.concatenate(fake), np.concatenate(cls)


def run(video, model_label, cascade, progress=gr.Progress()):
    if not video:
        raise gr.Error('Hãy tải lên một video hoặc chọn một video mẫu.')
    run_name = MODELS[model_label]
    model, cfg, _, device = get_model(run_name)
    size, thr = cfg['image_size'], cfg['threshold']
    progress(0, desc='Đọc video')
    frames, fps = read_video(video)
    if not frames:
        raise gr.Error('Không đọc được video.')
    progress(0.15, desc='Phát hiện khuôn mặt')
    boxes = []
    for k in range(0, len(frames), 32):
        # video thường chỉ có một người: lấy mặt lớn nhất như với video thật lúc train
        boxes += [preprocess.pick_face(b, None)
                  for b in preprocess.detect_faces(frames[k:k + 32])]
    found = [i for i, b in enumerate(boxes) if b is not None]
    if not found:
        raise gr.Error('Không tìm thấy khuôn mặt nào trong video.')
    progress(0.45, desc='Mô hình dự đoán')
    crops = [preprocess.square_crop(frames[i], boxes[i], SCALE, size, None) for i in found]
    masks, fake, cls = predict(crops, model, device, size)
    # kết luận dùng xác suất gốc (như evaluate.py); hiển thị dùng bản làm trơn 5 frame
    # để giảm nhấp nháy giữa các frame liền kề
    video_fake = float(fake.mean())
    masks_s = uniform_filter1d(masks, SMOOTH, axis=0, mode='nearest')
    fake_s = uniform_filter1d(fake, SMOOTH, mode='nearest')

    area = np.full(len(frames), np.nan)
    fake_t = np.full(len(frames), np.nan)
    out_path = os.path.join(tempfile.mkdtemp(), 'ket_qua.webm')
    h, w = frames[0].shape[:2]
    # OpenCV pip không ghi được H.264; VP8/WebM thì trình duyệt phát được
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'VP80'), fps, (w, h))
    idx = {i: k for k, i in enumerate(found)}
    for i, f in enumerate(progress.tqdm(frames, desc='Ghi video kết quả')):
        f = f.copy()
        if i in idx:
            k = idx[i]
            fake_t[i] = fake_s[k]
            # nối tầng: frame bị phân loại là thật thì không tô vùng nào
            show = not cascade or fake_s[k] >= 0.5
            area[i] = (masks_s[k] > thr).mean() if show else 0
            x1, y1, x2, y2 = boxes[i].astype(int)
            box_color = COLOR if fake_s[k] >= 0.5 else (80, 175, 76)
            if show:
                mask = paste_back(masks_s[k], boxes[i], f.shape) > thr
                f[mask] = (0.5 * f[mask] + 0.5 * np.array(COLOR)).astype(np.uint8)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(f, contours, -1, COLOR, 3)
            cv2.rectangle(f, (x1, y1), (x2, y2), box_color, 2)
            # OpenCV không vẽ được chữ có dấu
            text = 'GIA {:.0%}'.format(fake_s[k]) if fake_s[k] >= 0.5 else \
                'THAT {:.0%}'.format(1 - fake_s[k])
            cv2.putText(f, text, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        box_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(f, 'khong thay mat', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(f)
    writer.release()

    t = np.arange(len(frames)) / fps
    df = pd.concat([
        pd.DataFrame({'Thời gian (giây)': t, 'Giá trị (%)': 100 * fake_t,
                      'Chỉ số': 'Xác suất giả (đầu phân loại)'}),
        pd.DataFrame({'Thời gian (giây)': t, 'Giá trị (%)': 100 * area,
                      'Chỉ số': 'Diện tích vùng tô (% ảnh mặt)'}),
    ]).dropna()
    gallery = [(panel(crops[k], masks_s[k], thr),
                'giây {:.1f} · xác suất giả {:.0%} · vùng tô {:.0%}'.format(
                    t[found[k]], fake_s[k], area[found[k]]))
               for k in pick_frames(fake_s)]
    is_fake = video_fake >= 0.5
    methods = cls.mean(0)
    method_label = {CLASS_NAMES[j]: float(methods[j]) for j in range(len(CLASS_NAMES))}
    stats = ('**{}** frame đã xử lý · **{}** frame có mặt · **{:.0%}** frame bị phân loại '
             'là giả (> 0,5)').format(len(frames), len(found), float((fake >= 0.5).mean()))
    return (verdict_card(video_fake, is_fake), method_label, stats, out_path, df, gallery,
            reliability(run_name))


def verdict_card(p, is_fake):
    color, bg, word = ('#b42318', '#fef3f2', 'GIẢ MẠO') if is_fake else \
        ('#067647', '#ecfdf3', 'THẬT')
    conf = p if is_fake else 1 - p
    note = ''
    if abs(p - 0.5) < 0.15:
        note = ('<div style="margin-top:8px;font-size:14px">⚠️ Xác suất gần ngưỡng 50%: '
                'kết luận kém chắc chắn.</div>')
    return f'''<div style="border:2px solid {color};background:{bg};color:#101828;
border-radius:12px;padding:16px 20px">
<div style="font-size:14px;opacity:.75">Kết luận cho cả video</div>
<div style="font-size:34px;font-weight:700;color:{color};line-height:1.2">{word}</div>
<div style="font-size:15px;margin:6px 0 8px">Xác suất giả trung bình:
<b>{p:.1%}</b> · độ chắc chắn {conf:.0%}</div>
<div style="height:10px;border-radius:5px;background:#e4e7ec;overflow:hidden">
<div style="height:100%;width:{100 * p:.1f}%;background:{color}"></div></div>
{note}</div>'''


def reliability(run_name):
    """Bảng độ chính xác mức video của chính mô hình này trên các tập test."""
    rows = []
    for name, f in (('FF++ test, nén nhẹ (c23)', 'eval_test'),
                    ('FF++ test, nén mạnh (c40)', 'eval_test_c40'),
                    ('DFD, deepfake nguồn khác', 'eval_dfd')):
        try:
            with open(f'runs/{run_name}/{f}.json', encoding='utf-8') as fh:
                c = json.load(fh)['metrics']['cls']['video']
        except (OSError, KeyError):
            continue
        rr = c.get('bin_real_recall')
        # acc cân bằng = trung bình 2 recall; các tập test có nhiều video giả hơn thật (FF++ ~1:4, DFD ~1:8)
        bal = '–' if rr is None else '{:.1%}'.format((rr + c['bin_recall']) / 2)
        rows.append('| {} | {} | {:.1%} | {} | {:.1%} |'.format(
            name, '–' if rr is None else '{:.1%}'.format(rr), c['bin_recall'], bal, c['bin_acc']))
    if not rows:
        return 'Chưa có kết quả đánh giá cho mô hình này (chạy `src/evaluate.py`).'
    return '\n'.join([
        '| Tập kiểm tra | Nhận đúng video thật | Nhận đúng video giả | Accuracy cân bằng '
        '| Accuracy thường |',
        '| --- | --- | --- | --- | --- |', *rows, '',
        'Mô hình học từ FF++ (video tin tức, 4 kiểu ghép/sửa mặt). Video do AI tạo trọn '
        'vẹn, filter làm đẹp, video điện thoại/webcam hoặc deepfake từ công cụ khác có thể '
        'bị bỏ sót hoặc kết luận sai (xem dòng DFD).'])


def pick_frames(score, n=6):
    """n frame cách đều, thay frame cuối bằng frame có xác suất giả cao nhất."""
    ks = list(np.linspace(0, len(score) - 1, n - 1).round().astype(int))
    ks.append(int(score.argmax()))
    return sorted(set(ks))


def panel(crop, prob, thr):
    """Ảnh mặt | bản đồ xác suất từng pixel (thang cố định 0-1) | vùng vượt ngưỡng."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    size = rgb.shape[0]
    heat = (PROB_CMAP(prob)[..., :3] * 255).astype(np.uint8)
    # viền xám để ô toàn 0 (mặt thật) không trông như ảnh trống
    cv2.rectangle(heat, (0, 0), (size - 1, size - 1), (195, 194, 183), 2)
    over = rgb.copy()
    m = prob > thr
    over[m] = (0.5 * over[m] + 0.5 * np.array(COLOR[::-1])).astype(np.uint8)
    gap = np.full((size, 6, 3), 255, np.uint8)
    return np.hstack([rgb, gap, heat, gap, over])


CSS = '.gradio-container {max-width: 1200px !important; margin: auto}'
with gr.Blocks(title='Phát hiện video giả mạo khuôn mặt') as app:
    gr.Markdown('# Phát hiện video giả mạo khuôn mặt\n'
                'Tải video lên (hoặc chọn video mẫu), bấm **Phân tích**. Mô hình kết luận '
                'video thật hay giả, đoán kiểu giả mạo và tô đỏ vùng mặt bị chỉnh sửa.')
    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            # ô tải file, không phải trình phát: Chrome không phát được MPEG-4 Part 2,
            # HEVC (iPhone), H.264 4:4:4 (FF++); OpenCV vẫn đọc được
            video_in = gr.File(label='Video đầu vào (mp4, mov, avi, ...)',
                               file_types=['video'], type='filepath')
            if EXAMPLES:
                gr.Examples([[p] for p in EXAMPLES], [video_in],
                            label='Video mẫu (tập test FF++: thật, Deepfakes, Face2Face, '
                                  'FaceSwap, NeuralTextures)',
                            example_labels=[os.path.basename(os.path.dirname(
                                os.path.dirname(os.path.dirname(p)))).replace(
                                'youtube', 'Thật') for p in EXAMPLES])
            model_in = gr.Dropdown(list(MODELS), value=next(iter(MODELS), None),
                                   label='Mô hình')
            cascade_in = gr.Checkbox(True, label='Nối tầng: chỉ tô vùng giả ở frame bị '
                                                 'phân loại là giả (giảm tô nhầm)')
            button = gr.Button('Phân tích', variant='primary', size='lg')
        with gr.Column(scale=5):
            verdict_out = gr.HTML('<div style="padding:16px;opacity:.6">Kết luận sẽ hiện '
                                  'ở đây sau khi phân tích.</div>')
            stats_out = gr.Markdown()
            method_out = gr.Label(label='Kiểu giả mạo dự đoán (trung bình các frame)',
                                  num_top_classes=5)
    with gr.Tabs():
        with gr.Tab('Video kết quả'):
            video_out = gr.Video(label='Khung xanh = frame thật, khung đỏ = frame giả, '
                                       'vùng phủ đỏ = vùng bị chỉnh sửa')
        with gr.Tab('Theo thời gian'):
            plot_out = gr.LinePlot(x='Thời gian (giây)', y='Giá trị (%)', color='Chỉ số',
                                   label='Xác suất giả và diện tích vùng tô theo thời gian',
                                   y_lim=[0, 100])
        with gr.Tab('Khung hình tiêu biểu'):
            gallery_out = gr.Gallery(
                label='Ảnh mặt | xác suất giả từng pixel (trắng = 0, xanh đậm = 1) | '
                      'vùng vượt ngưỡng (đỏ)', columns=2, height='auto',
                object_fit='contain')   # mặc định 'cover' cắt ảnh 3 ô, chỉ còn ô giữa
        with gr.Tab('Độ tin cậy của mô hình'):
            reliab_out = gr.Markdown('Chọn mô hình và phân tích một video để xem.')
    button.click(run, [video_in, model_in, cascade_in],
                 [verdict_out, method_out, stats_out, video_out, plot_out, gallery_out,
                  reliab_out])

if __name__ == '__main__':
    app.launch(inbrowser=True, css=CSS)   # tự mở trình duyệt mặc định của Windows
