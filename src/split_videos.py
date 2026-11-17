"""Tách video FF++ thành thư mục train/val/test theo split chính thức (data/splits/*.json),
giữ nguyên cấu trúc thư mục của FF++ bên trong mỗi split; DFD (actors + DeepFakeDetection)
để nguyên bộ ở thư mục dfd (thử chéo); thêm vào đó DFD chia theo diễn viên (14 train / 14 test,
xem temporal.dfd_part) được bổ sung vào train và test.

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

# DFD chia theo diễn viên (temporal.dfd_part, seed 0) bổ sung vào train / test; video giả ghép
# mặt giữa diễn viên train và test chỉ nằm trong dfd (tránh rò rỉ danh tính)
from temporal import dfd_part   # noqa: E402
count = {}
for rel, sub in [('original_sequences/actors', KINDS),
                 ('manipulated_sequences/DeepFakeDetection', KINDS + ['masks'])]:
    for v in sorted(os.listdir(os.path.join(SRC, rel, 'c23', 'videos'))):
        split = dfd_part(v[:-4])
        if split is None:
            continue
        assert all(place(f'{rel}/{k}/videos/{v}') for k in sub), v
        count[split, rel.split('/')[1]] = count.get((split, rel.split('/')[1]), 0) + 1
for (split, group), c in sorted(count.items()):
    print(f'DFD → {split}: {group} {c} video')
