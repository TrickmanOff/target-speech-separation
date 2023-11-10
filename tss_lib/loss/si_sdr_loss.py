from typing import Dict

import torch
from torch import Tensor
from torch import nn


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


class SI_SDR_Loss(nn.Module):
    def __init__(self, mid_weight: float = 0.1, long_weight: float = 0.1):
        super().__init__()
        assert mid_weight >= 0 and long_weight >= 0 and mid_weight + long_weight <= 1
        self.coefs = {
            'w1': 1 - mid_weight - long_weight,
            'w2': mid_weight,
            'w3': long_weight,
        }

    def forward(self, targets: Tensor, **batch) -> Dict[str, Tensor]:
        """
        targets: (batch_dim, 1, T)
        waves: {'wi': (batch_dim, 1, T)} for i = 1, 2, 3
        """
        si_sdrs = {
            key: calc_si_sdr(targets, batch[key])
            for key in ['w1', 'w2', 'w3']
        }
        total_loss = sum(
            si_sdrs[key] * self.coefs[key] for key in si_sdrs
        )

        res = {
            f'{key}_SI-SDR': si_sdr
            for key, si_sdr in si_sdrs.items()
        }
        res['SI-SDR'] = total_loss
        return res
