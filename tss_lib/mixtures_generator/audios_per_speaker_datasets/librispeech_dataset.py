import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from tss_lib.config_processing.parse_config import ConfigParser
from tss_lib.mixtures_generator.audios_per_speaker_datasets.audios_per_speaker_indexer import AudiosPerSpeakerIndexer

# from tss_lib.base.base_dataset import BaseDataset
# from tss_lib.utils import ROOT_PATH
# from tss_lib.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechDataset(AudiosPerSpeakerIndexer):
    def __init__(self, part, config_parser: ConfigParser, data_dir=None, index_dir=None, *args, **kwargs):
        assert part in URL_LINKS or part == 'train_all'

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        if isinstance(index_dir, str):
            index_dir = Path(index_dir)

        if data_dir is None:
            data_dir = config_parser.get_data_root_dir() / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self._index_dir = data_dir if index_dir is None else index_dir
        if part == 'train_all':
            index = sum([self._get_or_load_index(part)
                         for part in URL_LINKS if 'train' in part], [])
        else:
            index = self._get_or_load_index(part)
        self._all_files_index = index

    def get_index(self) -> Dict[str, List[Path]]:
        """
        path structure: ".../{speaker_id}/{text_segment_id}/{speaker_id}-{text_segment_id}-*.flac
        """
        audios_per_speaker_index: Dict[str, List[Path]] = defaultdict(list)
        for data_dict in tqdm(self._all_files_index, f"Iterating over Librispeech index to get files for each speaker"):
            audio_path = Path(data_dict["path"])
            speaker_id = audio_path.name.split('-')[0]
            audios_per_speaker_index[speaker_id].append(audio_path)
        return dict(audios_per_speaker_index)

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self, part):
        """
        index format:
        [
            {"path": ..., "text": ..., "audio_len": ...},
            ...
        ]
        """
        index_path = self._index_dir / f"{part}_index.json"

        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs.add(dirpath)
        for flac_dir in tqdm(
                list(flac_dirs), desc=f"Preparing librispeech folders: {part}"
        ):
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.trans.txt"))[0]
            with trans_path.open() as f:
                for line in f:
                    f_id = line.split()[0]
                    f_text = " ".join(line.split()[1:]).strip()
                    flac_path = flac_dir / f"{f_id}.flac"
                    t_info = torchaudio.info(str(flac_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(flac_path.absolute().resolve()),
                            "text": f_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index
