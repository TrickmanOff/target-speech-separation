from abc import abstractmethod
from typing import Dict, List

from torch import Tensor, nn


class BaseLoss(nn.Module):
    @abstractmethod
    def get_loss_parts_names(self) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def forward(self, **batch) -> Dict[str, Tensor]:
        raise NotImplementedError()
