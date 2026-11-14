"""Tải ảnh mặt + mask đã cắt sẵn (FF++ c23, c40) từ Kaggle về đúng chỗ mà train/evaluate đọc.

Chạy: python download_data.py            (cả c23 và c40, khoảng 20 GB)
      python download_data.py --only c23  (chỉ c23, khoảng 10 GB)

Cần trước:
  1. Được chủ dataset chia sẻ (dataset riêng tư: giấy phép FF++ không cho phát tán công khai).
  2. pip install kaggle, và token Kaggle của bạn: Kaggle → Settings → API → Create New Token,
     đặt file vào ~/.kaggle/kaggle.json (Windows: C:\\Users\\<tên>\\.kaggle\\kaggle.json).

Kết quả:  c23 → data/faces/, data/index/{train,val,test}.csv
          c40 → data/faces_c40/, data/index_c40/
Chạy lại an toàn: bản nào đã có data/index*/train.csv thì bỏ qua.
Không có DFD (chỉ dùng để thử chéo): cần tải video gốc bằng download_all.ps1 rồi chạy preprocess.py.
"""
import argparse
import glob
import os
import shutil
import subprocess
import zipfile

OWNER = 'chituok'
DATASETS = {'c23': ('ffpp-faces-c23', 'data/faces', 'data/index'),
            'c40': ('ffpp-faces-c40', 'data/faces_c40', 'data/index_c40')}


def extract(tmp, faces_dir, index_dir):
    """Giải nén mọi zip trong tmp (Kaggle có thể gói thêm một lớp zip), rồi chuyển
    faces/ và index/ về đúng chỗ."""
    while True:
        zips = glob.glob(os.path.join(tmp, '**', '*.zip'), recursive=True)
        if not zips:
            break
        for z in zips:
            print('giải nén', z)
            with zipfile.ZipFile(z) as f:
                f.extractall(os.path.dirname(z))   # extractall tự bỏ đường dẫn tuyệt đối và '..'
            os.remove(z)   # xoá ngay để không tốn gấp đôi dung lượng
    index = glob.glob(os.path.join(tmp, '**', 'index', 'train.csv'), recursive=True)
    assert index, f'không thấy index/train.csv trong {tmp}: dataset tải về không đúng cấu trúc'
    root = os.path.dirname(os.path.dirname(index[0]))
    for name, dst in (('faces', faces_dir), ('index', index_dir)):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(os.path.join(root, name), dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', choices=list(DATASETS), help='chỉ tải một mức nén')
    args = p.parse_args()
    for comp in [args.only] if args.only else list(DATASETS):
        slug, faces_dir, index_dir = DATASETS[comp]
        if os.path.isfile(os.path.join(index_dir, 'train.csv')):
            print(f'{comp}: đã có {index_dir}, bỏ qua')
            continue
        tmp = os.path.join('data', '_download_' + comp)
        os.makedirs(tmp, exist_ok=True)
        print(f'{comp}: tải {OWNER}/{slug} ...')
        r = subprocess.run(['kaggle', 'datasets', 'download', f'{OWNER}/{slug}', '-p', tmp])
        if r.returncode:
            raise SystemExit(f'Tải lỗi. Kiểm tra token Kaggle và bạn đã được chia sẻ {OWNER}/{slug} chưa.')
        extract(tmp, faces_dir, index_dir)
        shutil.rmtree(tmp)
        print(f'{comp}: xong → {faces_dir}, {index_dir}')


if __name__ == '__main__':
    main()
