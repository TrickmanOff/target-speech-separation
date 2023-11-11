from typing import Dict, List

from torch import Tensor

from tss_lib.loss import CrossEntropyLoss, SI_SDR_Loss
from tss_lib.loss.base_loss import BaseLoss


class SpexPlusLoss(BaseLoss):
    def __init__(self, cls_loss_weight: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.cls_loss = CrossEntropyLoss(**kwargs)
        self.si_sdr_loss = SI_SDR_Loss(**kwargs)
        self.cls_loss_weight = cls_loss_weight  # gamma

    def get_loss_parts_names(self) -> List[str]:
        parts_names = self.cls_loss.get_loss_parts_names() + self.si_sdr_loss.get_loss_parts_names()
        parts_names.append('SI-SDR')
        if self.training:
            parts_names.append('cls_loss')
        return parts_names

    def forward(self, **batch) -> Dict[str, Tensor]:
        res_loss = {}
        si_sdr_loss: Dict[str, Tensor] = self.si_sdr_loss(**batch)
        res_loss.update(si_sdr_loss)
        if self.training:
            cls_loss: Tensor = self.cls_loss(**batch)
            res_loss['loss'] = (1 - self.cls_loss_weight) * si_sdr_loss['loss'] + self.cls_loss_weight * cls_loss
            res_loss['cls_loss'] = cls_loss
        else:
            res_loss['loss'] = res_loss['loss'] * (1 - self.cls_loss_weight)
        res_loss['SI-SDR'] = -si_sdr_loss['loss']
        return res_loss
