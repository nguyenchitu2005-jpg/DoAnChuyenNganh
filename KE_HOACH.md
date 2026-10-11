# Kế hoạch đồ án: Định vị vùng giả mạo deepfake trên khuôn mặt bằng pixel segmentation

Cập nhật: 11/10/2026 · Thời gian thực hiện: 12/10/2026 – 24/01/2027 (15 tuần)

Đề tài giải bài toán **pixel segmentation**: với mỗi ảnh mặt cắt từ frame video, mô hình dự đoán cho **từng pixel** xác suất pixel đó thuộc vùng bị chỉnh sửa. Đề tài so sánh 3 kiến trúc là U-Net, DeepLabV3+ và SegFormer-B0. Dữ liệu train và test là FaceForensics++ (FF++) c23 có mask, còn DeepFakeDetection (DFD) dùng để kiểm tra chéo. Theo yêu cầu của giảng viên, đề tài **không làm phân loại thật/giả** cho cả video.

## 1. Bài toán ở mức pixel

| | Mô tả |
| --- | --- |
| Đầu vào | Ảnh mặt RGB 256×256, cắt từ một frame video |
| Đầu ra | Bản đồ xác suất 256×256. Lấy ngưỡng 0,5 được mask nhị phân, trong đó 1 là pixel bị chỉnh sửa |
| Nhãn | Mask chính thức của FF++/DFD, cắt cùng khung với ảnh. Mặt thật dùng mask đen toàn bộ |
| Loss | Dice + BCE. Dice bù cho việc pixel giả ít hơn pixel thật, BCE giữ cho mặt thật được dự đoán đúng là đen |

Hình dạng mask khác nhau theo phương pháp giả, nên khi phân tích kết quả cần nêu rõ:

| Phương pháp | Loại giả | Hình dạng mask |
| --- | --- | --- |
| Deepfakes | Thay mặt (autoencoder) | Hình chữ nhật quanh mặt |
| FaceSwap | Thay mặt (đồ hoạ) | Theo viền mặt, chừa lỗ ở miệng |
| Face2Face | Tái hiện biểu cảm | Ảnh render 3D xám có bóng đổ, phủ gần cả mặt |
| NeuralTextures | Tái hiện biểu cảm (neural rendering) | Ảnh render xám, phủ gần cả mặt |
| DFD | Thay mặt (Google) | Gần chữ nhật quanh mặt |

Trên khung hình gốc, vùng giả chỉ chiếm khoảng 1–9% diện tích (đo trên 8 video mỗi loại). Sau khi cắt mặt, tỉ lệ này tăng lên. Con số thực tế sẽ được đo sau khi tiền xử lý xong.

## 2. Dữ liệu

Chỉ dùng dữ liệu tải từ server TUM bằng script chính thức của FF++ (server EU2), mức nén c23.

| Phần | Số video | Dung lượng | Dùng để |
| --- | --- | --- | --- |
| original (youtube) | 1000 | ~8,5 GB (cùng 4 phương pháp) | Mặt thật FF++ |
| Deepfakes, Face2Face, FaceSwap, NeuralTextures | 4 × 1000 | (tính chung ở trên) | Mặt giả FF++ |
| Mask 4 phương pháp | 4 × 1000 | 44 MB / 14 GB / 15 GB / 12 GB | Nhãn pixel |
| DFD giả c23 + mask | 3068 + 3068 | 20 GB + 3,9 GB | Kiểm tra chéo |
| DFD thật (actors) c23 | 363 | 2,8 GB | Đo báo nhầm khi kiểm tra chéo |
| Bản **c40** (nén mạnh) của original, 4 phương pháp, DFD giả và thật | 5000 + 3068 + 363 | (đang tải) | Test độ bền với video nén mạnh. Dùng chung mask c23 vì cùng frame |

- **Chia tập:** dùng split chính thức của FF++ (`data/splits/{train,val,test}.json`), gồm 360/70/70 cặp, tức 720/140/140 video gốc. Mỗi cặp `[a, b]` cho ra 2 video gốc và 2 video giả mỗi phương pháp (`a_b`, `b_a`). Như vậy một người chỉ xuất hiện trong một tập, không chia ngẫu nhiên theo frame.
- **DFD** không dùng để train, chỉ dùng để đo khả năng tổng quát.
- **FaceShifter** không có mask nên không dùng.
- Mỗi video gốc có 4 video giả tương ứng, nên khoảng 80% frame là giả. Vì vậy khi train cần lấy mẫu cân bằng (mục 4).

## 3. Tiền xử lý (`src/preprocess.py`)

| Bước | Cách làm |
| --- | --- |
| 1. Lấy frame | 32 frame cách đều mỗi video. Đọc video và video mask **song song từng frame**, nên chỉ số frame luôn khớp |
| 2. Phát hiện mặt | RetinaFace ResNet-50 (`batch-face`, chạy GPU), ngưỡng tin cậy 0,9. Phát hiện trên ảnh thu nhỏ về cạnh dài 960 rồi đổi khung về ảnh gốc |
| 3. Chọn mặt | Video thật: mặt lớn nhất. Video giả: mặt **chứa nhiều vùng mask nhất**, vì cảnh DFD có thể có 2 người mà người lớn hơn không bị giả |
| 4. Cắt | Khung vuông có cạnh bằng 1,3 lần cạnh dài của mặt. Phần ra ngoài ảnh tô đen. Cắt ảnh và mask bằng **cùng một khung** |
| 5. Nhị phân mask | Ngưỡng **> 10**, lưu dạng 0/255. Mask Face2Face/FaceSwap/NeuralTextures là ảnh xám có bóng đổ: ngưỡng 127 làm mất 10–20% vùng giả |
| 6. Resize | 256×256. Ảnh dùng INTER_AREA khi thu nhỏ, INTER_LINEAR khi phóng to. Mask dùng INTER_NEAREST để biên không bị nhoè |
| 7. Lưu | PNG (không thêm nhiễu nén) vào `data/faces/<nhóm>/{frames,masks}/<video>/<chỉ số frame>.png`. Nhật ký mỗi video ghi vào `data/faces/log.jsonl` |

Chạy thử trên 21 video: khung cắt và mask khớp nhau. Toàn bộ 8.431 video mất khoảng 3 tiếng (3 tiến trình trên RTX 3050).

## 4. Pipeline huấn luyện

| Thành phần | File | Nội dung |
| --- | --- | --- |
| Danh sách mẫu | `src/build_index.py` | `data/index/{train,val,test,dfd}.csv`. Mỗi dòng là một frame: đường dẫn ảnh, mask, phương pháp, video, nhãn |
| Dataset | `src/dataset.py` | Trả về ảnh (3,256,256) và mask (1,256,256) 0/1. Frame thật dùng mask đen |
| Augment | `src/dataset.py` | Lật ngang, xoay ±10°, scale 0,9–1,1, dịch ±5%: áp cho **cả ảnh và mask**. Nén JPEG 40–100, blur, noise, độ sáng: **chỉ áp cho ảnh** |
| Cân bằng | `balanced_sampler` | Mỗi batch trung bình **50% frame thật, 50% frame giả**. Đã kiểm tra: 49% qua 50 batch |
| Train | `src/train.py` | AMP float16, AdamW, OneCycleLR, batch 16, 10 epoch. Lưu `last.pt` mỗi epoch, log TensorBoard |
| Chọn checkpoint | `src/train.py` | `best.pt` theo điểm **(IoU ảnh giả + (1 − tỉ lệ báo nhầm)) / 2** |
| Đánh giá | `src/evaluate.py` | Chạy trên val, test hoặc dfd. In bảng theo từng phương pháp, lưu `eval_<split>.json` |

Không chọn checkpoint chỉ theo IoU. Lần chạy thử (1 epoch, 5% dữ liệu) cho IoU ảnh giả 0,78 nhưng **82% mặt thật bị báo nhầm**. Mask Face2Face/NeuralTextures phủ gần cả mặt, nên một mô hình chỉ biết "tô khuôn mặt" vẫn đạt IoU cao.

## 5. Mô hình

| Mô hình | Ý tưởng | Encoder (pretrained ImageNet) | Tham số | Learning rate |
| --- | --- | --- | --- | --- |
| U-Net | Encoder-decoder, skip connection giữ chi tiết biên | EfficientNet-B0 | 6,3M | 3e-4 |
| DeepLabV3+ | Tích chập giãn (ASPP) nắm bối cảnh nhiều tỉ lệ | EfficientNet-B0 | 4,9M | 3e-4 |
| SegFormer-B0 | Transformer nhẹ, quan hệ toàn cục giữa các vùng | MiT-B0 (`nvidia/mit-b0`) | 3,7M | 1e-4 |

Cả ba dùng cùng dữ liệu, ảnh 256×256, loss, số epoch và cách chọn checkpoint, để so sánh công bằng. Cấu hình nằm trong `configs/{unet,deeplabv3plus,segformer}.yaml`. Ước tính dựa trên lần chạy thử (dữ liệu cùng kích thước): mỗi epoch khoảng 32 phút, 10 epoch khoảng 5,5–6 tiếng mỗi mô hình. Chạy ở 224×224 batch 8 chỉ tốn 0,66 GB VRAM.

## 6. Đánh giá ở mức pixel (`src/metrics.py`)

| Chỉ số | Cách tính | Ý nghĩa |
| --- | --- | --- |
| IoU | Trung bình theo ảnh giả: giao / hợp của mask dự đoán và mask thật | Chỉ số chính |
| Dice | Trung bình theo ảnh giả: 2 × giao / tổng diện tích | Thường dùng trong paper |
| Pixel Precision, Recall, F1 | Cộng dồn mọi pixel của ảnh giả | Cho biết mô hình tô thừa hay tô thiếu |
| Pixel-AUC | 256 pixel ngẫu nhiên mỗi ảnh, không phụ thuộc ngưỡng | So sánh 3 mô hình |
| Tỉ lệ báo nhầm | Phần trăm ảnh mặt thật có vùng dự đoán giả lớn hơn 1% diện tích | IoU không đo được trên mask đen |

Báo cáo các bảng sau:
1. FF++ test, từng phương pháp.
2. DFD (kiểm tra chéo).
3. Độ bền: mô hình train trên c23, test trên **c40** (cắt mặt bằng cùng `preprocess.py`, dùng mask c23), và trên ảnh c23 bị nén JPEG nhiều mức, blur, noise.

## 7. Lệnh chạy

```
# 1. Tải dữ liệu (tự chờ khi server TUM sập)
powershell -ExecutionPolicy Bypass -File download_all.ps1
python watch_progress.py

# 2. Tiền xử lý: cắt mặt và mask
.venv\Scripts\python src/preprocess.py

# 3. Danh sách mẫu và kiểm tra dữ liệu
.venv\Scripts\python src/build_index.py
.venv\Scripts\python src/check_data.py --n 200

# 4. Train (thử nhanh trước bằng --fraction 0.05 --epochs 1)
.venv\Scripts\python src/train.py --config configs/unet.yaml
tensorboard --logdir runs

# 5. Đánh giá
.venv\Scripts\python src/evaluate.py runs/unet_effb0/best.pt --split test
.venv\Scripts\python src/evaluate.py runs/unet_effb0/best.pt --split dfd
```

## 8. Lịch thực hiện

| Tuần | Thời gian | Công việc | Đầu ra |
| --- | --- | --- | --- |
| 1 | 12/10 – 18/10 | Tải dữ liệu TUM; tiền xử lý RetinaFace; kiểm tra frame/mask | `data/faces/` hoàn chỉnh |
| 2 | 19/10 – 25/10 | Index, Dataset, augment, lấy mẫu cân bằng, metrics, vòng train/eval; chạy thử 5% dữ liệu | Pipeline chạy được |
| 3 | 26/10 – 01/11 | Train U-Net EfficientNet-B0 | Trọng số U-Net |
| 4 | 02/11 – 08/11 | Đánh giá U-Net: FF++ test từng phương pháp, DFD, tỉ lệ báo nhầm | Bảng kết quả U-Net |
| 5 | 09/11 – 15/11 | Train DeepLabV3+ cùng cấu hình | Trọng số DeepLabV3+ |
| 6 | 16/11 – 22/11 | Đánh giá như tuần 4 | Bảng kết quả DeepLabV3+ |
| 7 | 23/11 – 29/11 | Test độ bền trên c40 và JPEG/blur/noise cho 2 mô hình; dự phòng trễ | Bảng độ bền |
| 8 | 30/11 – 06/12 | Giữa kỳ: so sánh U-Net và DeepLabV3+, trình bày tiến độ | Báo cáo giữa kỳ |
| 9 | 07/12 – 13/12 | Train SegFormer-B0 | Trọng số SegFormer |
| 10 | 14/12 – 20/12 | Đánh giá SegFormer, kể cả độ bền | Bảng kết quả SegFormer |
| 11 | 21/12 – 27/12 | So sánh 3 kiến trúc; phương pháp nào khó định vị nhất; hình mask dự đoán và mask thật; ca sai | Bảng so sánh, hình minh hoạ |
| 12 | 28/12 – 03/01 | Demo Gradio: RetinaFace cắt mặt **giống lúc train**, dự đoán mask, dán ngược lên video, biểu đồ diện tích vùng giả theo thời gian | Demo chạy được |
| 13 | 04/01 – 10/01 | Viết chương 1–3: tổng quan, cơ sở lý thuyết segmentation, dữ liệu và phương pháp | Bản nháp chương 1–3 |
| 14 | 11/01 – 17/01 | Viết chương 4–5: thực nghiệm, kết luận; dọn code, viết README | Báo cáo và mã nguồn hoàn chỉnh |
| 15 | 18/01 – 24/01 | Slide, video demo dự phòng, tập trình bày | Slide và bản trình bày |

Nếu chậm tiến độ: dùng tuần 7 để bù trước. Nếu vẫn trễ thì bỏ SegFormer và so sánh 2 kiến trúc CNN. Phần train có thể chuyển lên Kaggle (GPU T4/P100) để nhanh hơn.

## 9. Tình trạng (11/10/2026)

- [x] Môi trường: Python 3.12 (`.venv`), PyTorch 2.14 CUDA 13.0, segmentation_models_pytorch, transformers, albumentations, batch-face
- [x] Tải toàn bộ dữ liệu TUM: 5.000 video c23 FF++, mask 4 phương pháp, DFD giả/thật và mask (75 GB)
- [ ] Tải bản c40 để test độ bền (đang tải)
- [x] Split chính thức FF++ (`data/splits/`)
- [x] Viết `preprocess.py`, thử trên 21 video
- [ ] Tiền xử lý toàn bộ 8.431 video (đang chạy)
- [x] Viết `build_index.py`, `dataset.py`, `metrics.py`, `models.py`, `train.py`, `evaluate.py`, `check_data.py`, 3 file config
- [ ] Chạy `build_index.py` và `check_data.py` trên dữ liệu mới, đo tỉ lệ vùng giả sau khi cắt
- [ ] Chạy thử U-Net trên 5% dữ liệu mới
- [ ] Train và đánh giá U-Net, DeepLabV3+, SegFormer-B0
- [ ] Test độ bền, phân tích, demo, báo cáo

## 10. Rủi ro và lưu ý

| Rủi ro | Cách xử lý |
| --- | --- |
| Server TUM hay sập hoặc chậm (sập khoảng 1 ngày, 10–11/10) | Dữ liệu đã tải đủ. `download_all.ps1` tự chờ server, script tải có thử lại và tải song song 8 luồng |
| GPU chỉ 4 GB | Mô hình nhẹ (3,7–6,3M tham số), AMP, batch 16; có thể chuyển lên Kaggle |
| IoU cao giả tạo trên Face2Face/NeuralTextures (mask phủ cả mặt) | Luôn báo IoU kèm tỉ lệ báo nhầm; chọn checkpoint theo điểm kết hợp |
| Demo cắt mặt khác lúc train làm kết quả sai | Dùng lại đúng hàm cắt của `preprocess.py` (RetinaFace, nới 1,3 lần, 256×256) |
| Video xem trên Windows báo lỗi 0x80004005 | Video là H.264 4:4:4; dùng VLC để xem, OpenCV để đọc trong code |
| Giấy phép dữ liệu | FF++ và DFD chỉ cho nghiên cứu: không chia sẻ dữ liệu, không tạo deepfake người chưa đồng ý |
| Demo không tuyệt đối | Hiển thị bản đồ xác suất và cảnh báo "có thể sai" |

## 11. Tài liệu tham khảo

- [FaceForensics++](https://github.com/ondyari/FaceForensics): Rössler và cộng sự, ICCV 2019 (gồm dữ liệu DeepFakeDetection của Google)
- [RetinaFace](https://arxiv.org/abs/1905.00641): Deng và cộng sự, CVPR 2020; bản PyTorch [batch-face](https://github.com/elliottzheng/batch-face)
- [U-Net](https://arxiv.org/abs/1505.04597): Ronneberger và cộng sự, MICCAI 2015
- [DeepLabV3+](https://arxiv.org/abs/1802.02611): Chen và cộng sự, ECCV 2018
- [SegFormer](https://arxiv.org/abs/2105.15203): Xie và cộng sự, NeurIPS 2021
- [EfficientNet](https://arxiv.org/abs/1905.11946): Tan và Le, ICML 2019
- [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch), [Hugging Face Transformers](https://github.com/huggingface/transformers)
