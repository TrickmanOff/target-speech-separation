from typing import List

import torch.nn.functional as F
from torch import Tensor

from tss_lib.loss.base_loss import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def get_loss_parts_names(self) -> List[str]:
        return []

    def forward(self, target_speaker_id: Tensor, speakers_log_probs: Tensor, **batch) -> Tensor:
        """
        target_speaker_id:  (batch_dim,)

        speakers_log_probs: (batch_dim, num_classes0
        """
        return F.nll_loss(speakers_log_probs, target_speaker_id, reduction='mean')
