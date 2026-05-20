from __future__ import annotations

from torch import nn

from .matchers import HSPMDABMMatcher, MultiScalePseudoSiameseMatcher, PseudoSiameseMatcher, SingleStreamMatcher


def build_model(
    model_name: str,
    dropout: float = 0.2,
    cbam_reduction: int = 16,
    attention_placement: str = "layer4",
    patch_sizes: tuple[int, ...] = (64, 128, 256),
    fusion_weights: tuple[float, ...] | None = None,
    final_backbone: str = "resnet34",
) -> nn.Module:
    common = {
        "dropout": dropout,
        "cbam_reduction": cbam_reduction,
        "attention_placement": attention_placement,
    }
    if model_name == "resnet18":
        return SingleStreamMatcher(backbone="resnet18", use_cbam=False, **common)
    if model_name == "resnet18_cbam":
        return SingleStreamMatcher(backbone="resnet18", use_cbam=True, **common)
    if model_name == "ps_resnet18":
        return PseudoSiameseMatcher(backbone="resnet18", use_cbam=False, **common)
    if model_name == "ps_resnet18_cbam":
        return PseudoSiameseMatcher(backbone="resnet18", use_cbam=True, **common)
    if model_name == "ps_resnet34_cbam":
        return PseudoSiameseMatcher(backbone="resnet34", use_cbam=True, **common)
    if model_name == "ps_resnet50_cbam":
        return PseudoSiameseMatcher(backbone="resnet50", use_cbam=True, **common)
    if model_name == "final_multiscale":
        return MultiScalePseudoSiameseMatcher(
            backbone=final_backbone,
            use_cbam=True,
            scales=patch_sizes,
            fusion_weights=fusion_weights,
            **common,
        )
    if model_name == "g10_hspm_dabm":
        return HSPMDABMMatcher(backbone=final_backbone, **common)
    raise ValueError(f"Unknown model_name: {model_name}")
