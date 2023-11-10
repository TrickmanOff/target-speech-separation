"""
handmade datasets of mixtures
"""
import json
import os
import shutil
from pathlib import Path

from speechbrain.utils.data_utils import download_file

from tss_lib.config_processing.parse_config import ConfigParser
from tss_lib.datasets.base_dataset import BaseDataset
from tss_lib.datasets.mixtures_storage import SimpleMixturesStorage
from tss_lib.utils.util import EnhancedJSONEncoder


URL_LINKS = {
    "dev-clean-3s": "https://www.googleapis.com/drive/v3/files/1FeRerEjCW0ibqP7EfvjgVpmxESmXJqAH?alt=media&key=AIzaSyBAigZjTwo8uh77umBBmOytKc_qjpTfRjI",
    "test-clean": "https://www.googleapis.com/drive/v3/files/10Yv8CThNaYTB55kW17a6LbtUDKXscxON?alt=media&key=AIzaSyBAigZjTwo8uh77umBBmOytKc_qjpTfRjI",
}


class LibrispeechDataset(BaseDataset):
    def __init__(self, part, config_parser: ConfigParser, data_dir=None, index_dir=None, *args, **kwargs):
        assert part in URL_LINKS or part == 'train_all'

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        if isinstance(index_dir, str):
            index_dir = Path(index_dir)

        if data_dir is None:
            data_dir = config_parser.get_data_root_dir() / "data" / "datasets" / "librispeech_mixtures"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self._index_dir = data_dir if index_dir is None else index_dir
        if part == 'train_all':
            index = sum([self._get_or_load_index(part)
                         for part in URL_LINKS if 'train' in part], [])
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, config_parser=config_parser, **kwargs)

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        os.remove(str(arch_path))

    def _get_or_load_index(self, part):
        index_path = self._index_dir / f"{part}_index.json"

        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2, cls=EnhancedJSONEncoder)
        return index

    def _create_index(self, part):
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        mix_storage = SimpleMixturesStorage(split_dir)
        return mix_storage.get_index()
