"""Tách video FF++ thành thư mục train/val/test theo split chính thức (data/splits/*.json),
giữ nguyên cấu trúc thư mục của FF++ bên trong mỗi split; DFD (actors + DeepFakeDetection)
để riêng ở thư mục dfd vì chỉ dùng để thử chéo.

Chạy: python src/split_videos.py   → data/ff_split/{train,val,test,dfd}/...
Dùng hard link (cùng ổ đĩa) nên không tốn thêm dung lượng; xoá thư mục này không ảnh
hưởng video gốc. Chạy lại an toàn: file đã có thì bỏ qua.
"""
import json
import os
import shutil

SRC, DST = 'data/FaceForensics', 'data/ff_split'
METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
KINDS = ['c23', 'c40']   # thư mục "masks" của bản manipulated cũng được tách theo


def place(rel):
    src, dst = os.path.join(SRC, rel), os.path.join(DST, split, rel)
    if not os.path.isfile(src) or os.path.exists(dst):
        return os.path.exists(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:   # khác ổ đĩa hoặc hệ thống file không hỗ trợ hard link
        shutil.copy2(src, dst)
    return True


for split in ['train', 'val', 'test']:
    with open(f'data/splits/{split}.json') as f:
        pairs = json.load(f)
    reals = sorted({v for p in pairs for v in p})
    fakes = sorted({f'{a}_{b}' for a, b in pairs} | {f'{b}_{a}' for a, b in pairs})
    n = sum(place(f'original_sequences/youtube/{k}/videos/{v}.mp4') for k in KINDS for v in reals)
    for m in METHODS:
        n += sum(place(f'manipulated_sequences/{m}/{k}/videos/{v}.mp4')
                 for k in KINDS + ['masks'] for v in fakes)
    print(f'{split}: {len(reals)} video thật, {len(fakes)} video giả mỗi phương pháp, {n} file')
    assert n == len(reals) * 2 + len(fakes) * len(METHODS) * 3, 'thiếu video gốc'

split = 'dfd'
n = 0
for rel, sub in [('original_sequences/actors', KINDS),
                 ('manipulated_sequences/DeepFakeDetection', KINDS + ['masks'])]:
    for k in sub:
        d = os.path.join(SRC, rel, k, 'videos')
        n += sum(place(f'{rel}/{k}/videos/{v}') for v in sorted(os.listdir(d)))
print(f'dfd: {n} file')
