from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from tss_lib.metric.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, pred_wave_index: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = ScaleInvariantSignalDistortionRatio()
        self.pred_wave_index = pred_wave_index

    def __call__(self, target_wave: Tensor, **batch) -> Tensor:
        pred_wave = batch[f'w{self.pred_wave_index}']
        return self.metric(pred_wave, target_wave).mean()
