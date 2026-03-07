"""
OSNet: Omni-Scale Network for Person Re-Identification.
========================================================

Minimal PyTorch implementation of OSNet (ICCV 2019, TPAMI 2021) for
feature extraction only.  No training code, no torchreid dependency.

Reference:
    Zhou et al. "Omni-Scale Feature Learning for Person Re-Identification."
    ICCV 2019.

Architecture: conv stem → 3 OSBlocks (multi-scale + channel gate) →
              global avg pool → 512-dim FC → L2-normalized features.

Pretrained weights (MSMT17, 4101 person identities) are downloaded
automatically from HuggingFace on first use.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_HF_REPO = "kaiyangzhou/osnet"
_WEIGHT_FILES = {
    "osnet_x1_0": "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    "osnet_x0_25": "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
}
_CONFIGS = {
    "osnet_x1_0": {"channels": [64, 256, 384, 512], "feature_dim": 512},
    "osnet_x0_25": {"channels": [16, 64, 96, 128], "feature_dim": 512},
}


class _ConvBnRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int, s: int = 1,
                 p: int = 0, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p,
                              bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class _Conv1x1(nn.Module):
    def __init__(self, in_c: int, out_c: int, s: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, stride=s, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class _Conv1x1Linear(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class _LightConv3x3(nn.Module):
    """Lightweight 3x3: pointwise 1x1 (linear) + depthwise 3x3 (nonlinear)."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False,
                               groups=out_c)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class _ChannelGate(nn.Module):
    """SE-style channel attention with 1/16 reduction."""

    def __init__(self, c: int, reduction: int = 16) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, c // reduction, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(c // reduction, c, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.sigmoid(self.fc2(self.relu(self.fc1(self.pool(x)))))
        return x * w


class _OSBlock(nn.Module):
    """Omni-scale block: 4 parallel branches at different depths + channel gate."""

    def __init__(self, in_c: int, out_c: int, br: int = 4) -> None:
        super().__init__()
        mid = out_c // br
        self.conv1 = _Conv1x1(in_c, mid)
        self.conv2a = _LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(_LightConv3x3(mid, mid),
                                    _LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(_LightConv3x3(mid, mid),
                                    _LightConv3x3(mid, mid),
                                    _LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(_LightConv3x3(mid, mid),
                                    _LightConv3x3(mid, mid),
                                    _LightConv3x3(mid, mid),
                                    _LightConv3x3(mid, mid))
        self.gate = _ChannelGate(mid)
        self.conv3 = _Conv1x1Linear(mid, out_c)
        self.downsample = (_Conv1x1Linear(in_c, out_c)
                           if in_c != out_c else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x1 = self.conv1(x)
        x2 = (self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1))
              + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1)))
        out = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(out + identity)


def _make_stage(n_blocks: int, in_c: int, out_c: int,
                pool: bool) -> nn.Sequential:
    layers: list[nn.Module] = [_OSBlock(in_c, out_c)]
    for _ in range(1, n_blocks):
        layers.append(_OSBlock(out_c, out_c))
    if pool:
        layers.append(nn.Sequential(_Conv1x1(out_c, out_c),
                                    nn.AvgPool2d(2, stride=2)))
    return nn.Sequential(*layers)


class OSNet(nn.Module):
    """OSNet feature extractor (inference only, 512-dim output)."""

    def __init__(self, channels: list[int], feature_dim: int = 512) -> None:
        super().__init__()
        self.conv1 = _ConvBnRelu(3, channels[0], 7, s=2, p=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = _make_stage(2, channels[0], channels[1], pool=True)
        self.conv3 = _make_stage(2, channels[1], channels[2], pool=True)
        self.conv4 = _make_stage(2, channels[2], channels[3], pool=False)
        self.conv5 = _Conv1x1(channels[3], channels[3])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build_osnet(variant: str = "osnet_x1_0",
                device: torch.device | None = None) -> OSNet:
    """Build OSNet and load MSMT17 person Re-ID pretrained weights.

    Weights are auto-downloaded from HuggingFace on first use (~18 MB).
    """
    if variant not in _CONFIGS:
        raise ValueError(f"Unknown OSNet variant: {variant}. "
                         f"Choose from {list(_CONFIGS.keys())}")

    cfg = _CONFIGS[variant]
    model = OSNet(channels=cfg["channels"], feature_dim=cfg["feature_dim"])

    weight_path = _download_weights(variant)
    _load_reid_weights(model, weight_path)

    if device is not None:
        model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("OSNet %s loaded  params=%.1fM  dim=%d  device=%s",
                variant, param_count, cfg["feature_dim"],
                device or "cpu")
    return model


def _download_weights(variant: str) -> str:
    """Download pretrained weights from HuggingFace (cached)."""
    try:
        from huggingface_hub import hf_hub_download
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "rws_reid")
        path = hf_hub_download(
            repo_id=_HF_REPO,
            filename=_WEIGHT_FILES[variant],
            cache_dir=cache_dir,
        )
        return path
    except Exception as e:
        raise RuntimeError(
            f"Failed to download OSNet weights. Install huggingface_hub: "
            f"pip install huggingface_hub. Error: {e}"
        ) from e


def _load_reid_weights(model: OSNet, path: str) -> None:
    """Load pretrained checkpoint, skipping classifier layer."""
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    own = model.state_dict()
    matched = OrderedDict()
    skipped: list[str] = []

    for k, v in state_dict.items():
        k_clean = k.removeprefix("module.")
        if k_clean in own and own[k_clean].shape == v.shape:
            matched[k_clean] = v
        else:
            skipped.append(k_clean)

    own.update(matched)
    model.load_state_dict(own)
    logger.info("Loaded %d/%d layers from Re-ID checkpoint (skipped %d: %s)",
                len(matched), len(state_dict), len(skipped),
                skipped[:5] if skipped else "none")
