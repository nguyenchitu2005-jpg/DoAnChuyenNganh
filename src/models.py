"""Kiến trúc segmentation; trả về logits (N,1,H,W). Baseline Swin-T + UPerNet: KE_HOACH.md mục 5."""
import torch.nn as nn


class SwinUperNet(nn.Module):
    """UPerNet + Swin-T của Hugging Face, khởi tạo từ checkpoint ADE20K (cả encoder và
    decoder); đầu phân lớp 150 lớp thay bằng 1 kênh. Logits đã phóng về kích thước ảnh vào."""

    def __init__(self, name='openmmlab/upernet-swin-tiny', pretrained=True):
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

    def forward(self, x):
        return self.net(pixel_values=x).logits


def build_model(arch, encoder=None, pretrained=True):
    if arch == 'swin_upernet':
        return SwinUperNet(encoder, pretrained)
    raise ValueError('unknown arch: ' + arch)
