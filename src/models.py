"""Kiến trúc segmentation; trả về logits (N,1,H,W). Baseline Swin-T + UPerNet: KE_HOACH.md mục 5.
B1 (đa nhiệm, mục 6): thêm đầu nhị phân và đầu 5 lớp, gọi model(x, with_cls=True)."""
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 5   # Real, Deepfakes, Face2Face, FaceSwap, NeuralTextures (= 5 phần tử đầu METHODS)


class SwinUperNet(nn.Module):
    """UPerNet + Swin-T của Hugging Face, khởi tạo từ checkpoint ADE20K (cả encoder và
    decoder); đầu phân lớp 150 lớp thay bằng 1 kênh. Logits đã phóng về kích thước ảnh vào.
    `multitask`: Global Average Pooling đặc trưng 1/32 của Swin rồi gắn đầu nhị phân
    (1 nút) và đầu đa lớp (5 nút)."""

    def __init__(self, name='openmmlab/upernet-swin-tiny', pretrained=True,
                 multitask=False):
        super().__init__()
        from transformers import UperNetConfig, UperNetForSemanticSegmentation
        # bỏ đầu phụ (auxiliary head) của ADE20K: kế hoạch chỉ dùng L_seg trên đầu chính
        kw = dict(num_labels=1, id2label={0: 'fake'}, label2id={'fake': 0},
                  use_auxiliary_head=False)
        if pretrained:
            self.net = UperNetForSemanticSegmentation.from_pretrained(
                name, ignore_mismatched_sizes=True, **kw)
        else:   # nạp trọng số từ checkpoint của mình ngay sau đó
            self.net = UperNetForSemanticSegmentation(UperNetConfig.from_pretrained(name, **kw))
        self.multitask = multitask
        if multitask:
            channels = self.net.backbone.channels[-1]
            self.bin_head = nn.Linear(channels, 1)
            self.cls_head = nn.Linear(channels, N_CLASSES)

    def forward(self, x, with_cls=False):
        if not with_cls:
            return self.net(pixel_values=x).logits
        # như UperNetForSemanticSegmentation.forward, giữ lại đặc trưng encoder cho 2 đầu
        feats = self.net.backbone.forward_with_filtered_kwargs(x).feature_maps
        seg = F.interpolate(self.net.decode_head(feats), size=x.shape[2:],
                            mode='bilinear', align_corners=False)
        pooled = feats[-1].mean((2, 3))
        return seg, self.bin_head(pooled)[:, 0], self.cls_head(pooled)


def build_model(arch, encoder=None, pretrained=True, multitask=False):
    if arch == 'swin_upernet':
        return SwinUperNet(encoder, pretrained, multitask)
    raise ValueError('unknown arch: ' + arch)
