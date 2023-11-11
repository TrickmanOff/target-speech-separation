from typing import Dict

import torch

from tss_lib.postprocessing.base_postprocessor import BasePostprocessor
from tss_lib.mixtures_generator.generator import normalize_wave_loudness


class LoudnessNormalizer(BasePostprocessor):
    def __init__(self, target_loudness: float, sr: int):
        self.target_loudness = target_loudness
        self.sr = sr

    def __call__(self, **batch) -> Dict:
        for wave_key in ['w1', 'w2', 'w3']:
            waves = batch[wave_key].detach().cpu().numpy()
            for i in range(batch[wave_key].shape[0]):
                waves[i, 0] = normalize_wave_loudness(waves[i, 0], self.target_loudness, self.sr)
            batch[wave_key] = torch.tensor(waves).to(batch[wave_key].device)
        return batch
