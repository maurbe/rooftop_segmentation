from typing import Dict, List, Tuple

import torch
import torch.nn as nn


SUPPORTED_LOSSES = ("dice", "iou", "focal", "pcc")


def _normalize_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in SUPPORTED_LOSSES:
        raise ValueError(
            f"Unsupported loss '{name}'. Supported: {', '.join(SUPPORTED_LOSSES)}"
        )
    return normalized


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation probabilities."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = float(eps)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.float()
        targets = targets.float()
        dims = tuple(range(1, preds.ndim))

        intersection = (preds * targets).sum(dim=dims)
        denom = preds.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class IoULoss(nn.Module):
    """Soft IoU (Jaccard) loss for binary segmentation probabilities."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = float(eps)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.float()
        targets = targets.float()
        dims = tuple(range(1, preds.ndim))

        intersection = (preds * targets).sum(dim=dims)
        union = (preds + targets - preds * targets).sum(dim=dims)
        iou = (intersection + self.eps) / (union + self.eps)
        return 1.0 - iou.mean()


class FocalLoss(nn.Module):
    """Binary focal loss on probability inputs in [0, 1]."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-7):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.float().clamp(min=self.eps, max=1.0 - self.eps)
        targets = targets.float()

        ce = -(targets * torch.log(preds) + (1.0 - targets) * torch.log(1.0 - preds))
        pt = targets * preds + (1.0 - targets) * (1.0 - preds)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()



class PCCLoss(nn.Module):
    """Pearson correlation coefficient loss for binary segmentation."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.contiguous().view(preds.size(0), -1).float()
        targets = targets.contiguous().view(targets.size(0), -1).float()

        weights = torch.ones_like(targets)

        valid_count = weights.sum(dim=1, keepdim=True)
        valid_count_safe = valid_count.clamp(min=1.0)

        preds_mean = (weights * preds).sum(dim=1, keepdim=True) / valid_count_safe
        targets_mean = (weights * targets).sum(dim=1, keepdim=True) / valid_count_safe

        preds_centered = (preds - preds_mean) * weights
        targets_centered = (targets - targets_mean) * weights

        numerator = (preds_centered * targets_centered).sum(dim=1)
        preds_norm = torch.sqrt((preds_centered**2).sum(dim=1) + self.eps)
        targets_norm = torch.sqrt((targets_centered**2).sum(dim=1) + self.eps)

        pcc = numerator / (preds_norm * targets_norm + self.eps)
        has_valid = valid_count.squeeze(1) > 0
        pcc = torch.where(has_valid, pcc, torch.ones_like(pcc))
        return (1.0 - pcc).mean()


def build_loss(loss_name_or_cfg, loss_cfg: Dict = None) -> nn.Module:
    """Build a segmentation loss object.

    Preferred usage:
      build_loss("dice", loss_cfg)

    Legacy usage (backward compatible):
      build_loss(loss_cfg)  # expects loss_cfg["name"] or defaults to dice
    """
    if loss_cfg is None:
        if not isinstance(loss_name_or_cfg, dict):
            raise TypeError(
                "When loss_cfg is omitted, the first argument must be a config dict."
            )
        loss_cfg = loss_name_or_cfg
        name = str(loss_cfg.get("name", loss_cfg.get("loss_train", "dice")))
    else:
        name = str(loss_name_or_cfg)

    name = _normalize_name(name)

    if name == "dice":
        eps = float(loss_cfg.get("dice_eps", loss_cfg.get("eps", 1e-7)))
        return DiceLoss(eps=eps)

    if name == "iou":
        eps = float(loss_cfg.get("iou_eps", loss_cfg.get("eps", 1e-7)))
        return IoULoss(eps=eps)

    if name == "focal":
        alpha = float(loss_cfg.get("focal_alpha", 0.25))
        gamma = float(loss_cfg.get("focal_gamma", 2.0))
        eps = float(loss_cfg.get("focal_eps", loss_cfg.get("eps", 1e-7)))
        return FocalLoss(alpha=alpha, gamma=gamma, eps=eps)

    if name == "pcc":
        eps = float(loss_cfg.get("eps", 1e-8))
        return PCCLoss(eps=eps)

    raise ValueError(f"Unsupported loss name: {name}")


def resolve_loss_config(loss_cfg: Dict) -> Tuple[str, List[str]]:
    """Resolve training/validation losses from TOML with legacy compatibility.

    Preferred schema:
      loss_train = "dice"
      loss_valid = ["iou", "focal"]

    Legacy schema fallback:
      name = "dice"
    """
    legacy = loss_cfg.get("name")

    train_name = loss_cfg.get("loss_train", legacy if legacy is not None else "dice")
    train_name = _normalize_name(str(train_name))

    valid_names_raw = loss_cfg.get("loss_valid")
    if valid_names_raw is None:
        valid_names = ["iou"]
    elif isinstance(valid_names_raw, str):
        valid_names = [valid_names_raw]
    else:
        valid_names = list(valid_names_raw)

    if len(valid_names) == 0:
        valid_names = ["iou"]

    valid_names = [_normalize_name(name) for name in valid_names]
    return train_name, valid_names


def compute_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Compute IoU metric for binary segmentation predictions."""
    preds_bin = (preds >= threshold).float()
    targets_bin = (targets >= threshold).float()

    intersection = (preds_bin * targets_bin).sum(dim=1)
    union = ((preds_bin + targets_bin) > 0).float().sum(dim=1)

    iou = torch.where(union > 0, intersection / union, torch.ones_like(intersection))
    return iou.mean()