"""3 kiến trúc segmentation được so sánh; cả ba trả về logits (N,1,H,W)."""
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F


class SegFormer(nn.Module):
    """SegFormer của Hugging Face; logits ra ở 1/4 kích thước, phóng lại bằng ảnh vào."""

    def __init__(self, name='nvidia/mit-b0'):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            name, num_labels=1, id2label={0: 'fake'}, label2id={'fake': 0},
            ignore_mismatched_sizes=True)

    def forward(self, x):
        logits = self.net(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear',
                             align_corners=False)


def build_model(arch, encoder='efficientnet-b0', pretrained=True):
    weights = 'imagenet' if pretrained else None
    if arch == 'unet':
        return smp.Unet(encoder, encoder_weights=weights, classes=1)
    if arch == 'deeplabv3plus':
        return smp.DeepLabV3Plus(encoder, encoder_weights=weights, classes=1)
    if arch == 'segformer':
        return SegFormer(encoder)
    raise ValueError('unknown arch: ' + arch)
