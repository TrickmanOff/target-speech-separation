import json
import os
from abc import abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Union


@dataclass
class MixtureMeta:
    target_speaker_id: Union[int, str] = None
    noise_speaker_id: Union[int, str] = None


class MixturesStorage:
    """
    A class that incapsulates the mixtures layout in memory
    """
    @abstractmethod
    def get_mix_filepaths(self, mix_id: str, with_ext: Optional[str] = None) -> Dict[str, Path]:
        """
        paths without extension if such file does not exist
        :return: {targets: ..., mix: ..., ref: ...}
        """
        raise NotImplementedError()

    @abstractmethod
    def add_mix_meta(self, mix_id: str, meta: MixtureMeta) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_mix_meta(self, mix_id: str) -> MixtureMeta:
        raise NotImplementedError()

    @abstractmethod
    def get_index(self) -> Dict[str, Dict]:
        """
        :return: {mix_id: {targets: ..., mix: ..., ref: ..., meta: ...}}
        """
        raise NotImplementedError()


# {mix_id}/{mix_id}-{type}.{ext}
class SimpleMixturesStorage(MixturesStorage):
    MIX_SUFFIX = 'mixed'
    TARGET_SUFFIX = 'target'
    REF_SUFFIX = 'ref'
    META_SUFFIX = 'meta.json'

    def __init__(self, dirpath: Union[str, Path]):
        self._dirpath = Path(dirpath)
        self._dirpath.mkdir(exist_ok=True)

    def _get_or_create_mix_dirpath(self, mix_id: str) -> Path:
        mix_dirpath = self._dirpath / mix_id
        mix_dirpath.mkdir(exist_ok=True, parents=True)
        return mix_dirpath

    def get_mix_filepaths(self, mix_id: str, with_ext: Optional[str] = None) -> Dict[str, Path]:
        with_ext = with_ext or ''
        mix_dirpath = self._get_or_create_mix_dirpath(mix_id)
        filenames = {suf: f'{mix_id}-{suf}{with_ext}' for suf in [self.MIX_SUFFIX, self.TARGET_SUFFIX, self.REF_SUFFIX]}
        for filename in os.listdir(mix_dirpath):
            part = os.path.splitext(filename)[0].split('-')[-1]
            if part in filenames:
                filenames[part] = filename
        return {
            'mix': mix_dirpath / filenames[self.MIX_SUFFIX],
            'target': mix_dirpath / filenames[self.TARGET_SUFFIX],
            'ref': mix_dirpath / filenames[self.REF_SUFFIX],
        }

    def add_mix_meta(self, mix_id: str, meta: MixtureMeta) -> None:
        meta_filepath = self._get_or_create_mix_dirpath(mix_id) / f'{mix_id}-{self.META_SUFFIX}'
        json.dump(asdict(meta), open(meta_filepath, 'w'))

    def get_mix_meta(self, mix_id: str) -> Union[MixtureMeta, None]:
        meta_filepath = self._get_or_create_mix_dirpath(mix_id) / f'{mix_id}-{self.META_SUFFIX}'
        if not meta_filepath.exists():
            return None
        return MixtureMeta(**json.load(open(meta_filepath, 'r')))

    def get_index(self) -> Dict[str, Dict]:
        index = {}
        for filename in os.listdir(self._dirpath):
            mix_dirpath = self._dirpath / filename
            if not os.path.isdir(mix_dirpath):
                continue
            mix_id = filename
            filepaths = self.get_mix_filepaths(mix_id)
            for filepath in filepaths.values():
                if not filepath.exists():
                    continue
            mix_data = {'meta': self.get_mix_meta(mix_id)}
            mix_data.update(filepaths)
            index[mix_id] = mix_data
        return index

# TODO: write another storage for testing
