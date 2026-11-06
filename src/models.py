"""Kiến trúc segmentation; trả về logits (N,1,H,W). Baseline Swin-T + UPerNet: KE_HOACH.md mục 5.
B1 (đa nhiệm, mục 6): thêm đầu nhị phân và đầu 5 lớp, gọi model(x, with_cls=True).
B2 (nhánh CNN, mục 6): EfficientNet chạy song song, hợp nhất với Swin ở 4 tỉ lệ."""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 5   # Real, Deepfakes, Face2Face, FaceSwap, NeuralTextures (= 5 phần tử đầu METHODS)


class Fuse(nn.Module):
    """concat(Swin, CNN) → Conv 1×1 → BN → ReLU về số kênh của Swin, cộng vào đặc trưng Swin.
    gamma của BN khởi tạo 0 nên lúc đầu đầu ra đúng bằng đặc trưng Swin: decoder UPerNet
    đã huấn luyện sẵn không bị phá bởi lớp hợp nhất còn ngẫu nhiên."""

    def __init__(self, swin_ch, cnn_ch):
        super().__init__()
        self.conv = nn.Conv2d(swin_ch + cnn_ch, swin_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(swin_ch)
        nn.init.zeros_(self.bn.weight)

    def forward(self, swin, cnn):
        return swin + F.relu(self.bn(self.conv(torch.cat([swin, cnn], 1))))


class SwinUperNet(nn.Module):
    """UPerNet + Swin-T của Hugging Face, khởi tạo từ checkpoint ADE20K (cả encoder và
    decoder); đầu phân lớp 150 lớp thay bằng 1 kênh. Logits đã phóng về kích thước ảnh vào.
    `multitask`: Global Average Pooling đặc trưng 1/32 rồi gắn đầu nhị phân (1 nút) và
    đầu đa lớp (5 nút). `cnn_branch`: tên mô hình timm (efficientnet_b4) lấy đặc trưng
    1/4, 1/8, 1/16, 1/32 để hợp nhất với Swin trước khi vào UPerNet."""

    def __init__(self, name='openmmlab/upernet-swin-tiny', pretrained=True,
                 multitask=False, cnn_branch=None):
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
        swin_channels = self.net.backbone.channels
        self.cnn = None
        if cnn_branch:
            import timm
            self.cnn = timm.create_model(cnn_branch, pretrained=pretrained,
                                         features_only=True, out_indices=(1, 2, 3, 4))
            self.fuse = nn.ModuleList(Fuse(s, c) for s, c in zip(
                swin_channels, self.cnn.feature_info.channels()))
        self.multitask = multitask
        if multitask:
            self.bin_head = nn.Linear(swin_channels[-1], 1)
            self.cls_head = nn.Linear(swin_channels[-1], N_CLASSES)

    def forward(self, x, with_cls=False):
        # như UperNetForSemanticSegmentation.forward, giữ lại đặc trưng encoder cho 2 đầu
        feats = self.net.backbone.forward_with_filtered_kwargs(x).feature_maps
        if self.cnn is not None:
            feats = [f(s, c) for f, s, c in zip(self.fuse, feats, self.cnn(x))]
        seg = F.interpolate(self.net.decode_head(feats), size=x.shape[2:],
                            mode='bilinear', align_corners=False)
        if not with_cls:
            return seg
        pooled = feats[-1].mean((2, 3))
        return seg, self.bin_head(pooled)[:, 0], self.cls_head(pooled)


def build_model(arch, encoder=None, pretrained=True, multitask=False, cnn_branch=None):
    if arch == 'swin_upernet':
        return SwinUperNet(encoder, pretrained, multitask, cnn_branch)
    raise ValueError('unknown arch: ' + arch)
