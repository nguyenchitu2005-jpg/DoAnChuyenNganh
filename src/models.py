"""Kiến trúc segmentation; trả về logits (N,1,H,W). Baseline Swin-T + UPerNet: KE_HOACH.md mục 5.
B1 (đa nhiệm, mục 6): thêm đầu nhị phân và đầu 5 lớp, gọi model(x, with_cls=True).
B2 (nhánh CNN, mục 6): EfficientNet chạy song song, hợp nhất với Swin ở 4 tỉ lệ.
B3 (nhánh biên, mục 6): đầu dự đoán biên vùng giả trên đặc trưng 1/4 của decoder,
gọi model(x, with_edge=True); logits biên đứng cuối bộ kết quả.
B4 (nhánh tần số, mục 6): bộ lọc SRM cố định + CNN nhỏ, hợp nhất với Swin ở 1/4 và 1/8."""
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


# 3 bộ lọc SRM lấy nhiễu dư (Zhou và cộng sự, CVPR 2018: nhánh nhiễu của RGB-N)
SRM_KERNELS = [
    [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
    [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2],
     [-1, 2, -2, 2, -1]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
]
SRM_SCALE = [4, 12, 2]


class SRMBranch(nn.Module):
    """3 bộ lọc SRM cố định trên từng kênh màu → 9 kênh nhiễu tần số cao, rồi một CNN nhỏ
    lấy đặc trưng 1/4 (64 kênh) và 1/8 (128 kênh)."""
    channels = (64, 128)

    def __init__(self):
        super().__init__()
        k = torch.tensor(SRM_KERNELS, dtype=torch.float32) / torch.tensor(SRM_SCALE)[:, None, None]
        # groups=3: kênh ra 3c..3c+2 là 3 bộ lọc trên kênh màu c; hằng số nên không lưu vào checkpoint
        self.register_buffer('srm', k.repeat(3, 1, 1)[:, None], persistent=False)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.to_4 = nn.Sequential(block(9, 32), block(32, 64))
        self.to_8 = block(64, 128)

    def forward(self, x):
        f4 = self.to_4(F.conv2d(x, self.srm.to(x.dtype), padding=2, groups=3))
        return f4, self.to_8(f4)


class SwinUperNet(nn.Module):
    """UPerNet + Swin-T của Hugging Face, khởi tạo từ checkpoint ADE20K (cả encoder và
    decoder); đầu phân lớp 150 lớp thay bằng 1 kênh. Logits đã phóng về kích thước ảnh vào.
    `multitask`: Global Average Pooling đặc trưng 1/32 rồi gắn đầu nhị phân (1 nút) và
    đầu đa lớp (5 nút). `cnn_branch`: tên mô hình timm (efficientnet_b4) lấy đặc trưng
    1/4, 1/8, 1/16, 1/32 để hợp nhất với Swin trước khi vào UPerNet. `edge_head`: đầu
    dự đoán biên (Conv 3×3 → BN → ReLU → Conv 1×1) trên đầu ra fpn_bottleneck của
    UPerNet (512 kênh, 1/4), ngay trước lớp phân lớp mask. `freq_branch`: 'srm' thêm
    nhánh SRMBranch, hợp nhất với đặc trưng 1/4 và 1/8 (sau nhánh CNN nếu có)."""

    def __init__(self, name='openmmlab/upernet-swin-tiny', pretrained=True,
                 multitask=False, cnn_branch=None, edge_head=False, freq_branch=None,
                 cls_pool='gap'):
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
        self.freq = None
        if freq_branch:
            assert freq_branch == 'srm', freq_branch
            self.freq = SRMBranch()
            self.freq_fuse = nn.ModuleList(Fuse(s, c) for s, c in zip(
                swin_channels[:2], SRMBranch.channels))
        self.multitask = multitask
        assert cls_pool in ('gap', 'mask'), cls_pool
        self.cls_pool = cls_pool
        if multitask:
            # 'mask' (C1): đặc trưng 1/8, 1/16, 1/32, mỗi tầng gom 2 cách (trung bình cả ảnh và
            # trung bình theo mask dự đoán), để vùng giả nhỏ như miệng không bị pha loãng
            dim = swin_channels[-1] if cls_pool == 'gap' else 2 * sum(swin_channels[1:])
            self.cls_norm = nn.LayerNorm(dim) if cls_pool == 'mask' else nn.Identity()
            self.bin_head = nn.Linear(dim, 1)
            self.cls_head = nn.Linear(dim, N_CLASSES)
        self.edge_head = None
        if edge_head:
            ch = self.net.decode_head.channels
            self.edge_head = nn.Sequential(
                nn.Conv2d(ch, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),
                nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))

    def forward(self, x, with_cls=False, with_edge=False):
        # như UperNetForSemanticSegmentation.forward, giữ lại đặc trưng encoder cho 2 đầu
        feats = self.net.backbone.forward_with_filtered_kwargs(x).feature_maps
        if self.cnn is not None:
            feats = [f(s, c) for f, s, c in zip(self.fuse, feats, self.cnn(x))]
        if self.freq is not None:
            feats = [f(s, c) for f, s, c in zip(self.freq_fuse, feats, self.freq(x))]                 + list(feats[2:])
        dec = {}
        if with_edge:   # lấy đặc trưng 1/4 của decoder mà không phải viết lại UperNetHead
            hook = self.net.decode_head.fpn_bottleneck.register_forward_hook(
                lambda module, inp, out: dec.update(feat=out))
        seg = F.interpolate(self.net.decode_head(feats), size=x.shape[2:],
                            mode='bilinear', align_corners=False)
        out = [seg]
        if with_cls:
            if self.cls_pool == 'gap':
                pooled = feats[-1].mean((2, 3))
            else:   # detach: loss phân loại không kéo lệch phần tô vùng
                m = torch.sigmoid(seg.detach().float())
                parts = []
                for f in feats[1:]:
                    f = f.float()
                    w = F.adaptive_avg_pool2d(m, f.shape[2:])
                    parts += [f.mean((2, 3)), (f * w).sum((2, 3)) / (w.sum((2, 3)) + 1e-4)]
                pooled = self.cls_norm(torch.cat(parts, 1))
            out +=[self.bin_head(pooled)[:, 0], self.cls_head(pooled)]
        if with_edge:
            hook.remove()
            out.append(F.interpolate(self.edge_head(dec['feat']), size=x.shape[2:],
                                     mode='bilinear', align_corners=False))
        return out[0] if len(out) == 1 else tuple(out)


def build_model(arch, encoder=None, pretrained=True, multitask=False, cnn_branch=None,
                edge_head=False, freq_branch=None, cls_pool='gap'):
    if arch == 'swin_upernet':
        return SwinUperNet(encoder, pretrained, multitask, cnn_branch, edge_head, freq_branch,
                           cls_pool)
    raise ValueError('unknown arch: ' + arch)
