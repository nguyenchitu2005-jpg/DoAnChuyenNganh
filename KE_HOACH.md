# Kế hoạch: Định vị vùng giả mạo khuôn mặt (Deepfake Segmentation) trên FaceForensics++

**Mô hình gốc (baseline):** Swin Transformer bản Tiny (Swin-T) làm encoder + UPerNet làm decoder
**Hướng phát triển:** bổ sung dần từng nhánh cải tiến, đo đóng góp của từng nhánh bằng thí nghiệm loại bỏ (ablation study)

---

## 0. Mục tiêu

| Cấp độ | Nhiệm vụ | Đầu ra |
|---|---|---|
| Chính | **Segment** vùng bị làm giả | Mask nhị phân theo từng pixel (1 = giả, 0 = thật) |
| Phụ | **Phân loại nhị phân** | Ảnh/video thật hay giả |
| Phụ | **Phân lớp đa lớp** | Phương pháp làm giả: Real, Deepfakes, Face2Face, FaceSwap, NeuralTextures |

**Đóng góp dự kiến của đề tài:**
1. Xây dựng baseline Swin-T + UPerNet cho định vị vùng giả trên FF++.
2. Phân tích hạn chế của baseline với bài toán deepfake.
3. Đề xuất mô hình cải tiến (nhánh CNN, nhánh biên, nhánh tần số, học đa nhiệm) và chứng minh bằng ablation.

---

## 1. Tổng quan lộ trình

| Giai đoạn | Nội dung | Kết quả cần có |
|---|---|---|
| 1 | Chuẩn bị dữ liệu | FF++ (c23, sau đó c40) kèm mask, chia train/val/test |
| 2 | Tiền xử lý | Ảnh mặt + mask đã cắt đồng bộ, file metadata |
| 3 | Baseline Swin-T + UPerNet | Bảng kết quả baseline |
| 4 | Cải tiến từng bước (B1 → B4) | Bảng ablation |
| 5 | Thí nghiệm bổ sung | So sánh decoder, mức nén, từng phương pháp |
| 6 | Suy luận trên video, demo | Video có tô vùng giả |
| 7 | Viết báo cáo | Báo cáo, slide bảo vệ |

---

## 2. Môi trường và công cụ

- **Ngôn ngữ, framework:** Python 3.10+, PyTorch 2.x
- **Mô hình:**
  - `transformers` (Hugging Face): `UperNetForSemanticSegmentation`, checkpoint `openmmlab/upernet-swin-tiny`
  - hoặc `mmsegmentation` (cấu hình UPerNet + Swin-T có sẵn)
  - `timm`: EfficientNet-B4 (cho nhánh CNN ở giai đoạn sau)
  - `segmentation_models_pytorch`: dùng cho thí nghiệm so sánh decoder
- **Xử lý ảnh, video:** OpenCV, `albumentations`
- **Phát hiện mặt:** RetinaFace (`insightface` hoặc `retinaface`) hoặc MTCNN (`facenet-pytorch`)
- **Đánh giá:** `torchmetrics`, `scikit-learn`
- **Theo dõi thí nghiệm:** TensorBoard hoặc Weights & Biases
- **Phần cứng gợi ý:** GPU từ 12 GB trở lên cho Swin-T segment ở 224×224; GPU nhỏ hơn thì giảm batch size, dùng mixed precision (AMP) và gradient accumulation

### Cấu trúc thư mục dự án

```
deepfake-seg/
├── configs/            # file cấu hình cho từng thí nghiệm (baseline, B1, B2...)
├── data/
│   ├── raw/            # video FF++ gốc, video giả, video mask
│   ├── processed/      # ảnh mặt và mask đã cắt
│   └── splits/         # train.json, val.json, test.json chính thức
├── src/
│   ├── preprocess/     # trích khung hình, cắt mặt, cắt mask
│   ├── datasets/       # Dataset, augmentation
│   ├── models/         # baseline, các nhánh cải tiến
│   ├── losses/         # BCE, Dice, loss biên
│   ├── train.py
│   ├── evaluate.py
│   └── infer_video.py
├── outputs/            # checkpoint, log, hình trực quan hóa
└── KE_HOACH.md
```

---

## 3. Giai đoạn 1: Chuẩn bị dữ liệu

### Việc cần làm
- [ ] Gửi yêu cầu và tải FaceForensics++ (bộ dữ liệu chỉ dùng cho nghiên cứu, cần tuân thủ điều khoản sử dụng).
- [ ] Tải các phần:
  - Video gốc (original sequences, 1000 video)
  - Video giả của 4 phương pháp: Deepfakes, Face2Face, FaceSwap, NeuralTextures
  - **Video mask** của từng phương pháp
- [ ] Bắt đầu với mức nén **c23**; tải thêm **c40** cho thí nghiệm mức nén.
- [ ] Tải file chia dữ liệu chính thức (720 / 140 / 140 video cho train / val / test).
- [ ] Kiểm tra: số lượng video, số khung hình, video mask khớp với video giả (cùng số khung, cùng kích thước).

### Lưu ý
- Mask mỗi phương pháp có hình dạng khác nhau: Deepfakes, FaceSwap, Face2Face thường phủ gần cả khuôn mặt; NeuralTextures chủ yếu quanh vùng miệng.
- Nếu phương pháp nào thiếu mask: tạo mask bằng hiệu ảnh giữa khung hình giả và khung hình thật tương ứng, lấy ngưỡng, làm mịn bằng phép toán hình thái học.

---

## 4. Giai đoạn 2: Tiền xử lý

### Các bước
1. **Trích khung hình:** mỗi video lấy 20–32 khung hình rải đều. Dùng **cùng chỉ số khung hình** cho video giả và video mask.
2. **Phát hiện mặt:** chạy RetinaFace trên khung hình, lấy khuôn mặt lớn nhất.
3. **Cắt mặt:** mở rộng hộp mặt 1,3 lần, cắt hình vuông. **Áp dụng đúng vùng cắt đó lên mask.**
4. **Resize:** ảnh 224×224 (khớp Swin-T window 7). Mask resize bằng nội suy *nearest*.
5. **Nhị phân hóa mask:** chọn ngưỡng (ví dụ > 127), kiểm tra bằng mắt.
6. **Ảnh thật:** gán mask toàn số 0.
7. **Lưu metadata** (CSV):

| Cột | Ý nghĩa |
|---|---|
| `img_path`, `mask_path` | Đường dẫn ảnh và mask |
| `label_bin` | 0 = thật, 1 = giả |
| `label_cls` | 0 = Real, 1 = Deepfakes, 2 = Face2Face, 3 = FaceSwap, 4 = NeuralTextures |
| `video_id`, `frame_idx` | Mã video, chỉ số khung hình |
| `method`, `compression` | Phương pháp, mức nén |
| `bbox` | Tọa độ vùng cắt (dùng để ghép mask về khung hình gốc) |

### Kiểm soát chất lượng
- [ ] Trực quan hóa ngẫu nhiên khoảng 50 cặp ảnh + mask chồng lên nhau, kiểm tra mask khớp đúng vị trí.
- [ ] Chia dữ liệu **theo video**, không theo khung hình (tránh rò rỉ dữ liệu).
- [ ] Xử lý mất cân bằng: số video giả gấp 4 lần video thật → lấy nhiều khung hình hơn từ video thật hoặc dùng `WeightedRandomSampler`.

---

## 5. Giai đoạn 3: Baseline Swin-T + UPerNet

### Kiến trúc
- **Encoder:** Swin-T, tiền huấn luyện ImageNet. Có thể khởi tạo cả encoder và decoder từ checkpoint UPerNet + Swin-T đã huấn luyện trên ADE20K để hội tụ nhanh hơn.
- **Decoder:** UPerNet (Pyramid Pooling Module + FPN) nhận đặc trưng 4 tỉ lệ (1/4, 1/8, 1/16, 1/32).
- **Đầu ra:** 1 kênh (logit mask), phóng to về kích thước ảnh đầu vào.

### Hàm mất mát
```
L_seg = BCE(mask_pred, mask_gt) + Dice(mask_pred, mask_gt)
```

### Cấu hình huấn luyện khởi đầu

| Tham số | Giá trị gợi ý |
|---|---|
| Kích thước ảnh | 224×224 |
| Optimizer | AdamW |
| Learning rate | 1e-4 (encoder có thể thấp hơn, ví dụ 5e-5) |
| Weight decay | 0,01–0,05 |
| Lịch learning rate | Warmup 1–2 epoch, sau đó cosine |
| Số epoch | 30–50, dừng sớm theo IoU trên tập val |
| Batch size | 16 (giảm nếu thiếu bộ nhớ, dùng gradient accumulation) |
| Mixed precision | Bật (AMP) |

### Tăng cường dữ liệu (áp dụng biến đổi hình học đồng thời cho ảnh và mask)
- Lật ngang
- Thay đổi nhẹ độ sáng, độ tương phản
- Nén JPEG ngẫu nhiên, làm mờ Gaussian, nhiễu nhẹ (giúp bền với mức nén)

### Kết quả cần có
- [ ] Bảng kết quả baseline trên c23: IoU, Dice, pixel-AUC
- [ ] Hình trực quan hóa mask dự đoán so với mask thật
- [ ] Phân tích lỗi: phương pháp nào kém nhất, lỗi thường gặp ở đâu (biên, vùng nhỏ...)

---

## 6. Giai đoạn 4: Cải tiến từng bước

Nguyên tắc: **mỗi lần chỉ thêm một thành phần**, giữ nguyên các thiết lập khác, để đo chính xác đóng góp của thành phần đó.

### B1. Học đa nhiệm (thêm đầu phân loại)
- **Mục đích:** để nhiệm vụ phân loại hỗ trợ việc học đặc trưng cho segment.
- **Cách làm:** lấy đặc trưng tầng cuối của encoder (1/32), Global Average Pooling, gắn 2 đầu:
  - Đầu nhị phân: 1 nút, BCE
  - Đầu đa lớp: 5 nút, Cross-Entropy
- **Hàm mất mát:**
  ```
  L = L_seg + λ1·L_bin + λ2·L_cls     (khởi đầu λ1 = λ2 = 0,5)
  ```
- **Đánh giá thêm:** Accuracy, AUC (nhị phân); F1 từng lớp, ma trận nhầm lẫn (đa lớp).

### B2. Thêm nhánh CNN (EfficientNet-B4)
- **Mục đích:** bù đắp chi tiết cấp thấp (viền ghép, kết cấu da) mà Swin làm mờ do chia patch 4×4.
- **Chọn B4 thay vì B0:** B0 (3,6 M tham số) quá nhỏ so với Swin-T; B4 (16,7 M) mạnh hơn cả Xception (mô hình chuẩn của FF++) với chi phí tương đương, và được dùng nhiều trong phát hiện deepfake. Đổi lại thời gian train tăng khoảng 30%.
- **Cách làm:**
  - EfficientNet-B4 (`timm`, `features_only=True`, trọng số ImageNet) chạy song song với Swin-T, lấy đặc trưng ở 4 tỉ lệ:
    - B4: 32, 56, 160, 448 kênh
    - Swin-T: 96, 192, 384, 768 kênh
  - Ở mỗi tỉ lệ: nối (concat) → Conv 1×1 → BatchNorm → ReLU, đưa về số kênh của Swin, rồi **cộng** vào đặc trưng Swin. gamma của BatchNorm khởi tạo 0 nên lúc đầu mô hình chạy y như B1, decoder UPerNet đã huấn luyện sẵn không bị phá.
  - Đặc trưng hợp nhất đưa vào UPerNet như cũ.
- **Thí nghiệm bổ sung (nếu còn quota):** so sánh loại nhánh CNN (EfficientNet-B0, Xception, ResNet-18) với mọi thiết lập khác giữ nguyên.
- **Mở rộng (tùy chọn):** thay concat bằng cross-attention giữa hai nhánh.

### B3. Thêm nhánh dự đoán biên
- **Mục đích:** làm sắc biên mask, tập trung vào viền ghép mặt (nơi lộ dấu vết rõ nhất).
- **Cách làm:**
  - Tạo nhãn biên từ mask thật bằng gradient hình thái học: `edge = dilate(mask) − erode(mask)` (kernel 3–5).
  - Gắn đầu dự đoán biên vào đặc trưng độ phân giải cao (1/4) của decoder.
  - Hàm mất mát: BCE có trọng số (vì pixel biên rất ít) hoặc Dice.
  ```
  L = L_seg + λ1·L_bin + λ2·L_cls + λ3·L_edge     (khởi đầu λ3 = 1,0)
  ```
- Ý tưởng tham khảo từ MVSS-Net và Face X-ray.

### B4. Thêm nhánh tần số
- **Mục đích:** tăng độ bền khi video bị nén nặng (c40).
- **Cách làm:**
  - Lọc ảnh qua 3 bộ lọc SRM cố định (hoặc biến đổi DCT theo khối 8×8).
  - Một CNN nhỏ trích đặc trưng từ ảnh đã lọc.
  - Hợp nhất vào các tỉ lệ 1/4 và 1/8 theo cách giống B2.
- **Đánh giá trọng tâm:** so sánh trên c40.

### B5. Tùy chọn (nếu còn thời gian)
- Thêm đặc trưng độ phân giải 1/2 vào decoder để tinh chỉnh mask.
- Thay Swin-T bằng Swin-S hoặc Swin V2.
- Làm mượt mask theo thời gian khi suy luận video (trung bình vài khung hình liên tiếp).

---

## 7. Bảng ablation (điền kết quả)

### Segment (mức nén c23)

| Mô hình | IoU | Dice | Pixel-AUC | Số tham số | Thời gian/ảnh |
|---|---|---|---|---|---|
| Baseline: Swin-T + UPerNet | 0,901 | 0,923 | 0,996 | 58,9 M | 29 ms (RTX 3050, batch 1, fp16) |
| + B1 (đa nhiệm) | | | | | |
| + B1 + B2 (nhánh CNN) | | | | | |
| + B1 + B2 + B3 (nhánh biên) | | | | | |
| + B1 + B2 + B3 + B4 (nhánh tần số) | | | | | |

### Phân loại (mức nén c23)

| Mô hình | Acc nhị phân | AUC nhị phân | Acc đa lớp | F1 macro đa lớp |
|---|---|---|---|---|
| Baseline + B1 | | | | |
| + B2 | | | | |
| + B3 | | | | |
| + B4 | | | | |

Lặp lại hai bảng trên cho **c40**.

---

## 8. Giai đoạn 5: Thí nghiệm bổ sung

- [ ] **So sánh decoder** (cùng encoder Swin-T): U-Net, U-Net++, DeepLabV3+, UPerNet.
- [ ] **So sánh encoder** (cùng decoder): EfficientNet-B0, ResNet-50, Swin-T.
- [ ] **Theo từng phương pháp làm giả:** IoU, Dice riêng cho Deepfakes, Face2Face, FaceSwap, NeuralTextures.
- [ ] **Theo mức nén:** c23 so với c40.
- [ ] **Kiểm thử chéo** (tùy chọn): đánh giá khả năng phân loại trên Celeb-DF v2 (bộ này chủ yếu dùng cho phân loại nếu không có mask).

---

## 9. Giai đoạn 6: Suy luận trên video và demo

1. Đọc video, trích khung hình.
2. Phát hiện và cắt mặt, lưu tọa độ vùng cắt.
3. Chạy mô hình: mask + nhãn thật/giả + phương pháp.
4. Ghép mask về khung hình gốc theo tọa độ đã lưu.
5. Làm mượt mask theo thời gian.
6. Xuất video có tô màu vùng giả, kèm nhãn và xác suất ở mức video (trung bình các khung hình).

---

## 10. Đánh giá và báo cáo

### Chỉ số
- **Segment:** IoU, Dice (F1 mức pixel), pixel-AUC
- **Nhị phân:** Accuracy, AUC (mức khung hình và mức video)
- **Đa lớp:** Accuracy, F1 từng lớp, F1 macro, ma trận nhầm lẫn

### Trực quan hóa cần có trong báo cáo
- Ảnh gốc | mask thật | mask dự đoán của baseline | mask dự đoán của mô hình cải tiến
- Biểu đồ đường loss và IoU qua các epoch
- Ma trận nhầm lẫn đa lớp
- Các trường hợp mô hình dự đoán sai và phân tích nguyên nhân

---

## 11. Tiến độ dự kiến (điều chỉnh theo thời gian thực tế)

| Tuần | Công việc |
|---|---|
| 1 | Tải dữ liệu, cài đặt môi trường, đọc tài liệu |
| 2–3 | Tiền xử lý, kiểm tra chất lượng mask |
| 4–5 | Baseline Swin-T + UPerNet, phân tích lỗi |
| 6 | B1: học đa nhiệm |
| 7 | B2: nhánh CNN |
| 8 | B3: nhánh biên |
| 9 | B4: nhánh tần số, thí nghiệm c40 |
| 10 | Thí nghiệm bổ sung (so sánh decoder, encoder, từng phương pháp) |
| 11 | Suy luận video, demo |
| 12 | Viết báo cáo, chuẩn bị slide |

---

## 12. Rủi ro và cách xử lý

| Rủi ro | Cách xử lý |
|---|---|
| Thiếu bộ nhớ GPU | Giảm batch size, bật AMP, gradient accumulation, gradient checkpointing |
| Mask lệch so với ảnh | Kiểm tra đồng bộ chỉ số khung hình và vùng cắt; trực quan hóa trước khi huấn luyện |
| Rò rỉ dữ liệu | Chia theo video theo split chính thức |
| Quá khớp (overfitting) | Tăng cường dữ liệu, weight decay, dừng sớm |
| Mất cân bằng lớp, mất cân bằng pixel | Lấy mẫu cân bằng; Dice loss; BCE có trọng số |
| NeuralTextures khó segment (vùng nhỏ) | Nhánh biên, attention, ảnh đầu vào lớn hơn |
| Swin huấn luyện không ổn định | Learning rate nhỏ, warmup, gradient clipping |
| Hết thời gian | Ưu tiên baseline + B1 + B2 + B3; B4, B5 là phần mở rộng |

---

## 13. Tài liệu tham khảo chính

1. Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*, ICCV 2019.
2. Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*, ICCV 2021.
3. Xiao et al., *Unified Perceptual Parsing for Scene Understanding* (UPerNet), ECCV 2018.
4. Tan & Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019.
5. Nguyen et al., *Multi-task Learning for Detecting and Segmenting Manipulated Facial Images and Videos*, BTAS 2019.
6. Li et al., *Face X-ray for More General Face Forgery Detection*, CVPR 2020.
7. Chen et al., *Image Manipulation Detection by Multi-View Multi-Scale Supervision* (MVSS-Net), ICCV 2021.
8. Zhou et al., *Learning Rich Features for Image Manipulation Detection* (nhánh nhiễu SRM), CVPR 2018.
