import logging
import random
from typing import Dict, Optional

import torchaudio
from torch.utils.data import Dataset

from tss_lib.config_processing.parse_config import ConfigParser
from tss_lib.datasets.mixtures_storage import MixtureMeta


logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(self, index: Dict[str, Dict], config_parser: ConfigParser,
                 limit: Optional[int] = None, *args, **kwargs):
        """
        :param limit: not more than `limit` random audios are taken from the index
        """
        # {mix_id: {targets: ..., mixed_wave: ..., ref_wave: ..., meta: ...}}
        index = [
            mix_data | {'mix_id': mix_id} for mix_id, mix_data in index.items()
        ]
        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        self._index = index
        self.config_parser = config_parser

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        mix_id = data_dict['mix_id']
        target_wave = self.load_audio(data_dict['target_wave']) if 'target_wave' in data_dict else None
        target_speaker_id = None
        noise_speaker_id = None
        if 'meta' in data_dict:
            meta: MixtureMeta = data_dict['meta'] if isinstance(data_dict['meta'], MixtureMeta) else MixtureMeta(**data_dict['meta'])
            target_speaker_id = meta.target_speaker_id
            # noise_speaker_id = meta.noise_speaker_id
        mixed_wave = self.load_audio(data_dict['mixed_wave'])
        ref_wave = self.load_audio(data_dict['ref_wave'])

        item = {
            'mix_id': mix_id,
            'mixed_wave': mixed_wave,
            'ref_wave': ref_wave
        }
        if target_wave is not None:
            item['target_wave'] = target_wave
        if target_speaker_id is not None:
            item['target_speaker_id'] = int(target_speaker_id)
        if noise_speaker_id is not None:
            item['noise_speaker_id'] = int(noise_speaker_id)

        return item

    def __len__(self) -> int:
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor
