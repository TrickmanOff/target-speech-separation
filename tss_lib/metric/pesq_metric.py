import torch
from pesq import NoUtterancesError
from torch import Tensor
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ

from tss_lib.metric.base_metric import BaseMetric


def calc_pesq(target_wave: Tensor, pred_wave: Tensor, sr: int = 16_000) -> Tensor:
    pesq_metric = PESQ(fs=sr, mode='wb')
    return pesq_metric(pred_wave, target_wave)


class PESQMetric(BaseMetric):
    def __init__(self, pred_wave_index: int, sr: int = 16_000, n_processes: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq_metric = PESQ(fs=sr, mode='wb', n_processes=n_processes)
        self.pred_wave_index = pred_wave_index

    def __call__(self, target_wave: Tensor, **batch) -> Tensor:
        pred_wave = batch[f'w{self.pred_wave_index}']
        try:
            return self.pesq_metric(pred_wave, target_wave).mean()
        except NoUtterancesError:
            print('NoUtterancesError')
            return torch.tensor(1.)
