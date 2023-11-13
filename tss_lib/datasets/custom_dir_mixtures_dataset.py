import logging
from pathlib import Path
from typing import Union

from tss_lib.datasets.base_dataset import BaseDataset
from tss_lib.datasets.mixtures_storage import CustomDirMixturesStorage

logger = logging.getLogger(__name__)


class CustomDirMixturesDataset(BaseDataset):
    def __init__(self, mixtures_dir: Union[str, Path], *args, **kwargs):
        mix_storage = CustomDirMixturesStorage(mixtures_dir)
        print('Creating an index for mixtures in the given directory...')
        mix_index = mix_storage.get_index()
        super().__init__(mix_index, *args, **kwargs)
