# Định vị vùng giả mạo deepfake trên khuôn mặt

Đồ án chuyên ngành: cho một video khuôn mặt, mô hình

1. **tô vùng bị chỉnh sửa** trên từng khung hình (segmentation theo pixel),
2. **kết luận video thật hay giả**,
3. **đoán phương pháp làm giả**: Deepfakes, Face2Face, FaceSwap hoặc NeuralTextures.

Dữ liệu train là FaceForensics++ (FF++), trộn hai mức nén c23 và c40. DeepFakeDetection (DFD) chỉ dùng để thử chéo trên deepfake từ nguồn khác.
Kế hoạch chi tiết và lý do chọn kiến trúc: [KE_HOACH.md](KE_HOACH.md).

## Các mô hình

Các mô hình được xây **cộng dồn**: B2 = B1 + nhánh CNN, B3 = B2 + nhánh biên, B4 = B3 + nhánh tần số. So sánh mỗi mô hình với mô hình ngay trước cho biết nhánh vừa thêm đóng góp gì (ablation). C1 là một nhánh rẽ riêng từ B1.

| Mô hình | Thêm gì | Tham số | ms/ảnh | Config |
|---|---|---|---|---|
| Baseline | Swin-T + UPerNet, chỉ tô vùng | 58.9M | 24 | `configs/swin_upernet_mix.yaml` |
| B1 | + đầu phân loại thật/giả và 5 lớp | 59.0M | 24 | `configs/b1_mix.yaml` |
| B2 | + nhánh CNN EfficientNet-B4 | 76.9M | 50 | `configs/b2_mix.yaml` |
| B3 | + đầu dự đoán biên vùng giả | 77.2M | 48 | `configs/b3_mix.yaml` |
| B4 | + nhánh tần số SRM | 77.6M | 49 | `configs/b4_mix.yaml` |
| C1 | B1 với đầu phân loại gom theo vùng giả | 59.0M | 26 | `configs/c1_mix.yaml` |

Mọi kiến trúc nằm trong [src/models.py](src/models.py); hàm loss nằm trong [src/train.py](src/train.py). Mọi số liệu trong phần này là trên tập test, theo cách đo ở phần [Kết quả trên tập test](#kết-quả-trên-tập-test).

### Baseline: Swin-T + UPerNet

**Cấu tạo:**
- **Encoder:** Swin Transformer bản Tiny. Ảnh mặt 224×224 đi qua 4 tầng, cho đặc trưng ở 4 tỉ lệ 1/4, 1/8, 1/16 và 1/32 của ảnh.
- **Decoder:** UPerNet trộn 4 tầng đặc trưng đó, gồm khối pyramid pooling trên tầng 1/32 và FPN đi từ trên xuống.
- **Đầu ra:** một bản đồ xác suất "giả" cho từng pixel. Các pixel có xác suất > 0.5 được tô.

**Khởi tạo:** cả encoder và decoder lấy trọng số đã huấn luyện sẵn của `openmmlab/upernet-swin-tiny` (segmentation ADE20K). Lớp cuối 150 lớp được thay bằng 1 kênh.

**Loss:** L_seg = BCE + Dice. BCE chấm điểm từng pixel. Dice chấm độ trùng của cả vùng, nên vùng giả nhỏ không bị lấn át bởi phần nền lớn.

**Kết quả:** tô vùng tốt trên FF++ (IoU c23 0.850). Nhưng mô hình không trực tiếp trả lời "video này thật hay giả", và thử chéo trên DFD chỉ đạt IoU 0.315.

### B1: học đa nhiệm (thêm đầu phân loại)

**Thêm vào:**
- Lấy trung bình toàn cục (Global Average Pooling) đặc trưng 1/32 của Swin, rồi nối hai lớp tuyến tính:
  - **đầu nhị phân** (1 nút): ảnh thật hay giả;
  - **đầu 5 lớp**: Thật, Deepfakes, Face2Face, FaceSwap, NeuralTextures.
- Loss: L = L_seg + 0.5·L_bin + 0.5·L_cls.
- Hai đầu này gần như không thêm tham số hay thời gian chạy.

**Tác dụng:**
- Mô hình **kết luận được thật/giả** cho cả video và **đoán được phương pháp làm giả**. Đây là phần demo dùng để ra kết luận.
- Hai nhiệm vụ phân loại buộc encoder học đặc trưng phân biệt thật/giả, nên **việc tô vùng cũng tốt lên**: IoU c23 0.850 → 0.873, c40 0.766 → 0.803, DFD 0.315 → 0.343.
- Cho phép **nối tầng**: ảnh mà đầu phân loại cho là thật thì không tô gì, giúp giảm tô nhầm trên mặt thật.

### B2: thêm nhánh CNN (EfficientNet-B4)

**Thêm vào:**
- EfficientNet-B4 (huấn luyện sẵn trên ImageNet) chạy song song với Swin trên cùng ảnh. Nó lấy đặc trưng ở đúng 4 tỉ lệ 1/4, 1/8, 1/16, 1/32.
- Ở mỗi tỉ lệ có một **khối hợp nhất**: ghép đặc trưng Swin với CNN, qua Conv 1×1 + BN + ReLU, rồi cộng vào đặc trưng Swin.
- Trọng số BN của khối hợp nhất khởi tạo bằng 0. Lúc bắt đầu train, khối này chưa đóng góp gì, nên decoder đã huấn luyện sẵn không bị phá.

**Lý do thêm:** Swin nhìn quan hệ toàn cục rất tốt. CNN giỏi bắt chi tiết cục bộ như đường viền ghép mặt, da bị làm mịn hay nhiễu lạ, là những dấu vết deepfake hay để lại.

**Tác dụng đo được (so với B1):**
- **Giảm tô nhầm trên mặt thật** rõ nhất trong các mô hình: tỉ lệ báo nhầm c23 10.9% → 4.8%.
- Nhưng IoU giảm (c23 0.873 → 0.826) và thử chéo DFD giảm mạnh (IoU 0.343 → 0.212).
- Số tham số tăng 30%, thời gian xử lý tăng gấp đôi (24 → 50 ms/ảnh).
- Kết luận: nhánh CNN **không đáng chi phí** trong cấu hình này.

### B3: thêm đầu dự đoán biên

**Thêm vào:**
- Một đầu nhỏ (Conv 3×3 → BN → ReLU → Conv 1×1) gắn vào đặc trưng 1/4 của decoder, ngay trước lớp tô vùng. Nó dự đoán **đường biên của vùng giả**.
- Nhãn biên được tạo tự động từ mask thật: phần chênh giữa mask nở ra và mask co lại, với kernel 5×5.
- Pixel biên chỉ chiếm vài phần trăm ảnh, nên loss biên là BCE có trọng số, để pixel biên không bị bỏ qua.
- Loss: L = L_B1 + 1.0·L_edge.
- Đầu biên chỉ dùng lúc train, không dùng khi suy luận.

**Lý do thêm:** đường ghép giữa mặt giả và phần còn lại là dấu vết quan trọng nhất của deepfake. Bắt decoder học vẽ biên giúp đường viền mask sắc hơn và hướng mô hình nhìn vào vùng ghép.

**Tác dụng đo được (so với B2):**
- IoU c23 gần như giữ nguyên (0.826 → 0.828).
- **Thử chéo DFD tốt lên:** IoU 0.212 → 0.287, AUC phân loại 0.722 → 0.835.
- Tỉ lệ báo nhầm c23 tăng 4.8% → 7.2%, IoU c40 giảm 0.748 → 0.717.

### B4: thêm nhánh tần số SRM

**Thêm vào:**
- **3 bộ lọc SRM cố định** (không học) áp lên từng kênh màu, cho 9 kênh "nhiễu dư". Đây là phần còn lại sau khi bỏ nội dung ảnh, chỉ giữ thành phần tần số cao. Bộ lọc lấy từ nhánh nhiễu của RGB-N (Zhou và cộng sự, CVPR 2018).
- Một CNN nhỏ (4 khối Conv) biến 9 kênh này thành đặc trưng ở 1/4 (64 kênh) và 1/8 (128 kênh).
- Hai tầng đặc trưng này được hợp nhất vào Swin bằng cùng loại khối hợp nhất như B2.

**Lý do thêm:** khi tạo deepfake, mặt giả được sinh ra rồi dán vào khung hình. Mắt thường khó thấy, nhưng vùng dán có thống kê nhiễu khác với phần ảnh gốc. SRM làm lộ ra sự khác biệt đó và ít phụ thuộc vào nội dung khuôn mặt hay cách làm giả cụ thể.

**Tác dụng đo được (so với B3):**
- Tốt lên ở mọi tập: IoU c23 0.828 → 0.848, c40 0.717 → 0.752, **DFD 0.287 → 0.403** (cao nhất trong các mô hình).
- Phân loại trên DFD tốt nhất: acc cân bằng 74.3%, AUC 0.853.
- Chỉ thêm khoảng 0.4M tham số.
- Kết luận: **nhánh tần số là nhánh giúp tổng quát hoá tốt nhất.**

### C1: B1 ưu tiên phân loại

**Thay đổi so với B1:**
- **Cách gom đặc trưng cho đầu phân loại.** B1 chỉ lấy trung bình cả ảnh ở tầng 1/32. Vùng giả nhỏ như miệng (NeuralTextures) dễ bị pha loãng.
  - C1 lấy đặc trưng ở 3 tầng 1/8, 1/16 và 1/32.
  - Mỗi tầng gom theo 2 cách: trung bình cả ảnh, và trung bình có trọng số theo **mask giả do chính mô hình dự đoán**.
  - Mask được tách gradient, nên loss phân loại không làm lệch phần tô vùng.
- **Trọng số loss phân loại** tăng từ 0.5 lên 1.0.
- **Chọn checkpoint theo acc cân bằng thật/giả trên val**, thay cho điểm tô vùng.

**Tác dụng (so với B1):**
- Phân loại trên FF++ c23 tốt nhất: acc cân bằng 93.8% → 94.9%, AUC 0.984 → 0.986.
- IoU c23 giảm 0.873 → 0.853, và DFD không tốt hơn B1.

## Kết quả trên tập test

Mọi mô hình đều train trộn c23 và c40. Thời gian xử lý đo với ảnh 224×224, batch 1, fp16, trên RTX 3050 Laptop.

**Segmentation, đo theo khung hình.** IoU tính trên ảnh giả. Tỉ lệ báo nhầm là phần trăm ảnh thật có vùng bị tô lớn hơn 1% diện tích ảnh.

| Mô hình | Tham số | ms/ảnh | IoU c23 | IoU c40 | IoU DFD | Báo nhầm c23 |
|---|---|---|---|---|---|---|
| Baseline | 58.9M | 24 | 0.850 | 0.766 | 0.315 | 10.4% |
| B1 | 59.0M | 24 | **0.873** | **0.803** | 0.343 | 10.9% |
| B2 | 76.9M | 50 | 0.826 | 0.748 | 0.212 | **4.8%** |
| B3 | 77.2M | 48 | 0.828 | 0.717 | 0.287 | 7.2% |
| B4 | 77.6M | 49 | 0.848 | 0.752 | **0.403** | 8.1% |
| C1 | 59.0M | 26 | 0.853 | 0.745 | 0.318 | 8.2% |

**Phân loại thật/giả, đo theo video.** Xác suất giả của một video là trung bình xác suất giả của các khung hình; video bị kết luận giả khi xác suất này lớn hơn 0.5. Tập test FF++ có 140 video thật và 558 video giả, DFD có 363 thật và 3068 giả. Vì số video thật và giả chênh nhau nhiều, bảng dùng **acc cân bằng**, tức trung bình của tỉ lệ nhận đúng video thật và tỉ lệ nhận đúng video giả.

| Mô hình | Acc cân bằng c23 | AUC c23 | Acc cân bằng c40 | AUC c40 | Acc cân bằng DFD | AUC DFD |
|---|---|---|---|---|---|---|
| B1 | 93.8% | 0.984 | **88.4%** | **0.956** | 69.5% | 0.807 |
| B2 | 93.6% | 0.981 | 88.1% | 0.950 | 61.9% | 0.722 |
| B3 | 93.5% | 0.984 | 86.6% | 0.935 | 68.5% | 0.835 |
| B4 | **95.0%** | 0.982 | 86.3% | 0.946 | **74.3%** | **0.853** |
| C1 | 94.9% | **0.986** | 87.3% | 0.947 | 68.8% | 0.774 |

**Nhận xét chính**

- Trên FF++, B1 và C1 tốt nhất mà không tăng thời gian chạy.
- B4 thử chéo trên DFD tốt nhất: IoU 0.403, AUC 0.853. Nhánh tần số SRM giúp mô hình tổng quát hơn.
- Nhánh EfficientNet (B2) làm thời gian xử lý tăng gấp đôi nhưng không cải thiện độ chính xác.
- Trên DFD, mọi mô hình nhận đúng trên 90% video thật nhưng chỉ bắt được 26–57% video giả. Mô hình thường bỏ sót deepfake từ nguồn khác FF++.

## Cài đặt

Cần Python 3.12 và GPU NVIDIA. Đánh giá và demo chạy được trên GPU 4 GB (RTX 3050 Laptop); train dùng GPU T4 16 GB trên Kaggle.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

## Dữ liệu

1. **Tải video FF++ và DFD.** Cần đăng ký với nhóm tác giả FF++ để được dùng script tải; bản script dùng trong đồ án là `faceforensics_download_v4.py`. [download_all.ps1](download_all.ps1) gọi script này cho mask và video, và tự chờ rồi tải tiếp khi server lỗi:
   ```powershell
   powershell -ExecutionPolicy Bypass -File download_all.ps1
   powershell -ExecutionPolicy Bypass -File download_all.ps1 -Compression c40
   ```
   Video được lưu vào `data/FaceForensics/`, giữ cấu trúc gốc của FF++. Chia train/val/test theo split chính thức trong `data/splits/{train,val,test}.json`.
2. **Cắt mặt và mask.** Mỗi video lấy 32 khung cách đều. Mặt được phát hiện bằng RetinaFace, nới khung 1.3 lần rồi đưa về 224×224:
   ```bash
   python src/preprocess.py                       # c23 -> data/faces
   python src/preprocess.py --compression c40 --out data/faces_c40 --boxes-from data/faces
   ```
   Bản c40 dùng lại đúng khung hình và khung cắt của bản c23. Xem tiến độ bằng `python preprocess_status.py`.
3. **Lập danh sách ảnh** cho từng split:
   ```bash
   python src/build_index.py
   python src/build_index.py --root data/faces_c40 --out data/index_c40 --compression c40
   ```
4. **(Tuỳ chọn) Tách video theo split** để lấy riêng video test, chẳng hạn cho demo. Script dùng hard link nên không tốn thêm dung lượng:
   ```bash
   python src/split_videos.py   # -> data/ff_split/{train,val,test,dfd}/
   ```

## Train và đánh giá

```bash
python src/train.py --config configs/b4_mix.yaml                     # train
python src/train.py --config configs/b4_mix.yaml --resume            # train tiếp từ last.pt
python src/train.py --config configs/b1_mix.yaml --fraction 0.05 --epochs 1   # chạy thử nhanh

python src/evaluate.py runs/b4_mix/best.pt --split test                    # FF++ c23
python src/evaluate.py runs/b4_mix/best.pt --split test --compression c40  # FF++ c40
python src/evaluate.py runs/b4_mix/best.pt --split dfd                     # thử chéo DFD
python src/evaluate.py runs/b4_mix/best.pt --split test --cascade          # nối tầng
```

- Kết quả train nằm trong `runs/<tên>/`: `best.pt`, `last.pt`, `history.jsonl` và `eval_*.json`.
- Với `--cascade`, ảnh mà đầu phân loại cho là thật sẽ không bị tô vùng nào. Cách này giảm tô nhầm trên ảnh thật.

**Các công cụ phân tích khác:**
- [src/visualize.py](src/visualize.py): so sánh mask dự đoán với mask thật, gồm ảnh ngẫu nhiên hoặc ảnh sai nhiều nhất.
- [src/robustness.py](src/robustness.py): đo độ bền khi ảnh bị nén JPEG, làm mờ hoặc thêm nhiễu.
- [src/analyze.py](src/analyze.py): phân tích lỗi theo nhóm.

### Train trên Kaggle

Mô hình trong đồ án được train trên Kaggle, mỗi phiên dùng 2 GPU T4.
1. Đóng gói ảnh mặt rồi tải lên Kaggle Dataset bằng `python src/pack_kaggle.py --user <tài khoản>`.
2. Chạy [kaggle/train_kaggle.ipynb](kaggle/train_kaggle.ipynb). Ô đầu tiên của notebook hướng dẫn cách cấu hình và cách train tiếp qua nhiều phiên 12 giờ.

## Demo

```bash
python src/demo.py   # mở http://127.0.0.1:7860
```

Tải lên một video hoặc chọn video mẫu thuộc tập test, rồi chọn mô hình. Demo trả về:
- kết luận thật hay giả, kèm xác suất;
- tỉ lệ dự đoán cho từng phương pháp làm giả;
- video kết quả có tô vùng giả;
- biểu đồ xác suất giả theo thời gian và các khung hình tiêu biểu;
- bảng độ tin cậy của mô hình trên tập test.

Demo cần thư mục `runs/<tên>/` chứa `best.pt`. Thư mục `runs/` không có trên GitHub.

**Giới hạn:** mô hình chỉ học từ FF++, gồm video tin tức và 4 kiểu làm giả. Với video quay bằng điện thoại, video có filter hoặc deepfake từ công cụ khác, mô hình dễ bỏ sót, như kết quả trên DFD đã cho thấy.

## Cấu trúc thư mục

```
configs/          cấu hình từng thí nghiệm (baseline, B1–B4, C1)
src/
  preprocess.py   cắt mặt và mask từ video
  build_index.py  danh sách ảnh theo split
  dataset.py      Dataset PyTorch và augmentation
  models.py       kiến trúc các mô hình
  train.py        train
  evaluate.py     đánh giá, nối tầng
  metrics.py      IoU, Dice, báo nhầm, acc, AUC
  demo.py         demo Gradio
  split_videos.py tách video theo split
  ...             visualize, robustness, analyze, pack_kaggle, check_data
kaggle/           notebook train trên Kaggle
data/, runs/      dữ liệu và checkpoint (không đưa lên GitHub)
```
