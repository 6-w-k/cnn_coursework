"""CIFAR-adapted ResNet-18 and controlled single-component variants.

Architecture reference: kuangliu/pytorch-cifar (MIT), retained under references/.
"""
import torch
from torch import nn

VARIANTS = ("baseline", "depthwise", "se", "gelu", "mish")


class DepthwiseSeparable(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        # Preserve the original block's BN/activation locations: no extra layers here.
        super().__init__(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )


class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, max(1, channels // reduction), 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(max(1, channels // reduction), channels, 1)

    def forward(self, x):
        return x * torch.sigmoid(self.fc2(self.relu(self.fc1(self.pool(x)))))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride, variant):
        super().__init__()
        act = {"gelu": nn.GELU, "mish": nn.Mish}.get(variant, nn.ReLU)
        conv = DepthwiseSeparable if variant == "depthwise" else self.standard_conv
        self.conv1 = conv(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = act()
        self.conv2 = conv(channels, channels, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.attention = SE(channels) if variant == "se" else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        self.act2 = act()

    @staticmethod
    def standard_conv(in_channels, channels, stride):
        return nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.attention(self.bn2(self.conv2(out)))
        return self.act2(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, variant="baseline", num_classes=10):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = {"gelu": nn.GELU, "mish": nn.Mish}.get(variant, nn.ReLU)()
        previous = 64
        for index, channels in enumerate((64, 128, 256, 512), 1):
            stage = nn.Sequential(
                BasicBlock(previous, channels, 1 if index == 1 else 2, variant),
                BasicBlock(channels, channels, 1, variant),
            )
            setattr(self, f"layer{index}", stage)
            previous = channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = stage(x)
        return self.linear(self.pool(x).flatten(1))


def model_statistics(model):
    """Conv/Linear MACs at [1,3,32,32]; FLOPs=2*MACs, excluding other operators."""
    macs = 0
    handles = []

    def count(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            macs += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(count))
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 3, 32, 32, device=next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "macs": macs, "flops": 2 * macs, "input_shape": [1, 3, 32, 32],
        "convention": "Conv2d/Linear MACs only; FLOPs=2*MACs. Excludes BN, activations, pooling, bias and elementwise attention/residual operations.",
    }
