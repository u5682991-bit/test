from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from .cbam import CBAM


RESNET_CHANNELS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}


def _make_resnet(name: str, in_channels: int) -> nn.Module:
    if name == "resnet18":
        model = models.resnet18(weights=None)
    elif name == "resnet34":
        model = models.resnet34(weights=None)
    elif name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    if in_channels != 3:
        old = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
    return model


class ResNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        in_channels: int = 3,
        use_cbam: bool = False,
        cbam_reduction: int = 16,
        attention_placement: str = "layer4",
    ) -> None:
        super().__init__()
        base = _make_resnet(backbone, in_channels)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.out_dim = RESNET_CHANNELS[backbone]
        self.use_cbam = use_cbam
        placements = {p.strip() for p in attention_placement.split(",") if p.strip()}
        self.cbam2 = CBAM(128 if backbone != "resnet50" else 512, cbam_reduction) if use_cbam and "layer2" in placements else nn.Identity()
        self.cbam3 = CBAM(256 if backbone != "resnet50" else 1024, cbam_reduction) if use_cbam and "layer3" in placements else nn.Identity()
        self.cbam4 = CBAM(self.out_dim, cbam_reduction) if use_cbam and "layer4" in placements else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.pool(x).flatten(1)
        return F.normalize(x, dim=1)


class SingleStreamMatcher(nn.Module):
    def __init__(self, backbone: str = "resnet18", use_cbam: bool = False, dropout: float = 0.2, **kwargs) -> None:
        super().__init__()
        self.feature = ResNetFeatureExtractor(backbone, in_channels=4, use_cbam=use_cbam, **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature.out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, sar: torch.Tensor, opt: torch.Tensor) -> torch.Tensor:
        x = torch.cat([sar, opt], dim=1)
        return self.head(self.feature(x))


class PseudoSiameseMatcher(nn.Module):
    def __init__(self, backbone: str = "resnet18", use_cbam: bool = False, dropout: float = 0.2, **kwargs) -> None:
        super().__init__()
        self.sar_branch = ResNetFeatureExtractor(backbone, in_channels=1, use_cbam=use_cbam, **kwargs)
        self.opt_branch = ResNetFeatureExtractor(backbone, in_channels=3, use_cbam=use_cbam, **kwargs)
        dim = self.sar_branch.out_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, sar: torch.Tensor, opt: torch.Tensor) -> torch.Tensor:
        fs = self.sar_branch(sar)
        fo = self.opt_branch(opt)
        fused = torch.cat([fs, fo, torch.abs(fs - fo), fs * fo], dim=1)
        return self.head(fused)


class MultiScalePseudoSiameseMatcher(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet34",
        use_cbam: bool = True,
        dropout: float = 0.2,
        scales: tuple[int, ...] = (64, 128, 256),
        fusion_weights: tuple[float, ...] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.branches = nn.ModuleList(
            [PseudoSiameseMatcher(backbone=backbone, use_cbam=use_cbam, dropout=dropout, **kwargs) for _ in scales]
        )
        if fusion_weights is None:
            fusion_weights = tuple([1.0 / len(scales)] * len(scales))
        weights = torch.tensor(fusion_weights, dtype=torch.float32)
        self.register_buffer("fusion_weights", weights / weights.sum())

    def forward(self, sar: torch.Tensor, opt: torch.Tensor) -> torch.Tensor:
        logits = []
        for size, branch in zip(self.scales, self.branches):
            sar_s = F.interpolate(sar, size=(size, size), mode="bilinear", align_corners=False)
            opt_s = F.interpolate(opt, size=(size, size), mode="bilinear", align_corners=False)
            logits.append(branch(sar_s, opt_s))
        stacked = torch.stack(logits, dim=0)
        return (stacked * self.fusion_weights.view(-1, 1, 1)).sum(dim=0)


class HSPMDABMMatcher(nn.Module):
    """G10 shared-weight dual-branch matcher for HSPM+DABM registration.

    SAR and Optical patches use lightweight modality adapters and then share one
    ResNet+CBAM encoder. The shared embedding is used both for patch-match
    training and for explicit correlation matrices during registration.
    """

    def __init__(self, backbone: str = "resnet34", dropout: float = 0.2, **kwargs) -> None:
        super().__init__()
        self.sar_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        self.opt_adapter = nn.Identity()
        self.shared_encoder = ResNetFeatureExtractor(backbone, in_channels=3, use_cbam=True, **kwargs)
        dim = self.shared_encoder.out_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim * 4 + 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def encode_sar(self, sar: torch.Tensor) -> torch.Tensor:
        return self.shared_encoder(self.sar_adapter(sar))

    def encode_opt(self, opt: torch.Tensor) -> torch.Tensor:
        return self.shared_encoder(self.opt_adapter(opt))

    def encode_pair(self, sar: torch.Tensor, opt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_sar(sar), self.encode_opt(opt)

    def score_embeddings(self, sar_features: torch.Tensor, opt_features: torch.Tensor) -> torch.Tensor:
        cosine = (sar_features * opt_features).sum(dim=1, keepdim=True)
        fused = torch.cat(
            [sar_features, opt_features, torch.abs(sar_features - opt_features), sar_features * opt_features, cosine],
            dim=1,
        )
        return self.head(fused)

    def forward(self, sar: torch.Tensor, opt: torch.Tensor) -> torch.Tensor:
        fs, fo = self.encode_pair(sar, opt)
        return self.score_embeddings(fs, fo)
