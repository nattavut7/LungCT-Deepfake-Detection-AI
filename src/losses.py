import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()

def make_loss(name, class_weights=None, gamma=2.0, device="cpu"):
    weight = None
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    name = name.lower()
    if name in ("ce", "cross_entropy"):
        return nn.CrossEntropyLoss()
    if name in ("weighted_ce", "weighted_cross_entropy"):
        return nn.CrossEntropyLoss(weight=weight)
    if name in ("focal", "focal_loss"):
        return FocalLoss(alpha=weight, gamma=gamma)
    raise ValueError(name)
