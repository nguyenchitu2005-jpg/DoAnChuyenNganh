"""Đóng gói ảnh mặt đã cắt + file index thành một file zip để tải lên Kaggle Dataset.

Chạy: python src/pack_kaggle.py
Mặc định lấy train, val, test (FF++ c23, khoảng 12 GB); DFD và c40 đánh giá ở máy nhà.
Kết quả: data/kaggle/ffpp_faces_c23.zip (bên trong: faces/..., index/<split>.csv)
và data/kaggle/dataset-metadata.json để tải lên bằng `kaggle datasets create`.
PNG đã nén sẵn nên zip chỉ lưu (không nén lại), đóng gói nhanh hơn.
"""
import argparse
import csv
import json
import os
import zipfile

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', default='data/faces')
    parser.add_argument('--index', default='data/index')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--out', default='data/kaggle')
    parser.add_argument('--slug', default='ffpp-faces-c23',
                        help='Tên dataset trên Kaggle (chữ thường, gạch nối).')
    parser.add_argument('--user', required=True,
                        help='Tên tài khoản Kaggle (username).')
    args = parser.parse_args()

    files = set()
    for split in args.splits:
        with open(os.path.join(args.index, split + '.csv'), newline='',
                  encoding='utf-8') as f:
            for r in csv.DictReader(f):
                files.add(r['frame'])
                if r['mask']:
                    files.add(r['mask'])

    os.makedirs(args.out, exist_ok=True)
    zip_path = os.path.join(args.out, 'ffpp_faces_c23.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as z:
        for split in args.splits:
            z.write(os.path.join(args.index, split + '.csv'),
                    'index/{}.csv'.format(split))
        for rel in tqdm(sorted(files), desc='zip'):
            z.write(os.path.join(args.root, rel), 'faces/' + rel)

    # Dataset riêng tư (mặc định của Kaggle): giấy phép FF++ không cho chia sẻ lại
    meta = {'title': 'FF++ faces c23 (private)',
            'id': '{}/{}'.format(args.user, args.slug),
            'licenses': [{'name': 'other'}]}
    with open(os.path.join(args.out, 'dataset-metadata.json'), 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print('{} files, {:.1f} GB -> {}'.format(
        len(files), os.path.getsize(zip_path) / 1e9, zip_path))


if __name__ == '__main__':
    main()
