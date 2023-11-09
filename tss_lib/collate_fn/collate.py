import logging
from typing import List

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


PADDING_VALUE = 0


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}

    """
    assuming the following:
    
    during training all mixes and targets lengths are the same:
    |mix_1| = |tgt_1| = |mix_2| = |tgt_2| = ... = |mix_n| = |tgt_n|
    
    during evaluation batch_size must be equal to 1, so collate_fn must do nothing
    
    thus, no padding for mix and target waves is used
    """

    # audio
    for wave_param in ['mixed_wave', 'ref_wave', 'target_wave']:
        if wave_param not in dataset_items[0]:
            assert wave_param == 'target_wave', f'No {wave_param} in batch'
            continue
        waves = [item[wave_param] for item in dataset_items]  # each wave of shape (1, time_dim)
        waves_length = torch.tensor([wave.shape[1] for wave in waves])
        if wave_param in ['mixed_wave', 'target_wave']:
            assert all(waves_length == waves_length[0]), f'Lengths of {wave_param}s are not the same in the batch:\n{waves_length}'
            waves_length = None
        else:
            # padding for references
            max_wave_len = waves_length.max()
            waves = [
                F.pad(wave, (0, max_wave_len - wave.shape[1]), value=PADDING_VALUE)
                for wave in waves
            ]
        result_batch[wave_param] = torch.concat(waves, dim=0)  # (batch_dim, time_dim)
        if waves_length is not None:
            result_batch[wave_param.removesuffix("wave") + 'length'] = waves_length

    # speakers' ids
    for speaker_id_param in ['target_speaker_id', 'noise_speaker_id']:
        if speaker_id_param not in dataset_items[0]:
            continue
        speakers_ids = [item[speaker_id_param] for item in dataset_items]
        result_batch[speaker_id_param] = torch.tensor(speakers_ids)

    # mixes' ids
    result_batch['mix_id'] = [item['mix_id'] for item in dataset_items]

    return result_batch
