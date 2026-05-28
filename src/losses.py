import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

class LogitAdjustedLoss(nn.Module):
    def __init__(self, class_priors: Iterable[float], tau: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.tau, self.reduction = tau, reduction
        priors = torch.tensor(class_priors, dtype=torch.float32).clamp_min(1e-6)
        priors = priors / priors.sum()
        self.register_buffer("log_prior", torch.log(priors))
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.tau * self.log_prior, targets, reduction=self.reduction)
