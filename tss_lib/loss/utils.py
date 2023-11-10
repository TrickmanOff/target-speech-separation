import torch
from torch import Tensor


def calc_si_sdr(target: Tensor, est: Tensor) -> Tensor:
    """
    targets: (batch_dim, 1, T) or (batch_dim, T)
    est:     (batch_dim, 1, T) or (batch_dim, T)
    """
    if target.dim() == 3:
        target = target.squeeze(1)
    if est.dim() == 3:
        est = est.squeeze(1)
    normed_target = (est * target).sum(axis=1, keepdim=True) / (target**2).sum(axis=1, keepdim=True) * target
    return 20 * torch.log10(torch.linalg.norm(normed_target, dim=1) / torch.linalg.norm(normed_target - est, dim=1))
