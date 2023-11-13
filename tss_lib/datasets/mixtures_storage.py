import json
import os
from abc import abstractmethod
from collections import defaultdict
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
        :return: {targets: ..., mixed_wave: ..., ref_wave: ...}
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
        :return: {mix_id: {targets: ..., mixed_wave: ..., ref_wave: ..., meta: ...}}
        """
        raise NotImplementedError()


# {mix_id}/{mix_id}-{type}.{ext}
class SimpleMixturesStorage(MixturesStorage):
    MIX_SUFFIX = 'mixed'
    TARGET_SUFFIX = 'target'
    REF_SUFFIX = 'ref'
    META_SUFFIX = 'meta.json'

    def __init__(self, dirpath: Union[str, Path], speakers_mapping: Optional[Dict[str, int]] = None):
        self._dirpath = Path(dirpath)
        self._dirpath.mkdir(exist_ok=True)
        self.speakers_mapping = speakers_mapping

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
            'mixed_wave': mix_dirpath / filenames[self.MIX_SUFFIX],
            'target_wave': mix_dirpath / filenames[self.TARGET_SUFFIX],
            'ref_wave': mix_dirpath / filenames[self.REF_SUFFIX],
        }

    def add_mix_meta(self, mix_id: str, meta: MixtureMeta) -> None:
        meta_filepath = self._get_or_create_mix_dirpath(mix_id) / f'{mix_id}-{self.META_SUFFIX}'
        if self.speakers_mapping:
            meta.target_speaker_id = self.speakers_mapping[meta.target_speaker_id]
            meta.noise_speaker_id = self.speakers_mapping[meta.noise_speaker_id]
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


# currently, can be used only for reading
# {mix_id}-{type}.{ext}
class CustomDirMixturesStorage(MixturesStorage):
    MIX_SUFFIX = 'mixed'
    TARGET_SUFFIX = 'target'
    REF_SUFFIX = 'ref'

    SUFFIXES = (MIX_SUFFIX, TARGET_SUFFIX, REF_SUFFIX)

    MIX_DIRNAME = 'mix'
    TARGET_DIRNAME = 'targets'
    REF_DIRNAME = 'refs'

    DIRNAMES = (MIX_DIRNAME, TARGET_DIRNAME, REF_DIRNAME)

    def __init__(self, dirpath: Union[str, Path]):
        self._dirpath = Path(dirpath)

    def get_mix_filepaths(self, mix_id: str, with_ext: Optional[str] = None) -> Dict[str, Path]:
        with_ext = with_ext or ''
        filedirs = {suf: self._dirpath / dirname for suf, dirname in zip(self.SUFFIXES, self.DIRNAMES)}
        filenames = {suf: f'{mix_id}-{suf}{with_ext}' for suf in self.SUFFIXES}
        for suf, filedir in filedirs.items():
            stem = filenames[suf]
            for filename in os.listdir(filedir):
                if os.path.splitext(filename)[0] == stem:
                    filenames[suf] = filename
                    break
        return {
            'mixed_wave': filedirs[self.MIX_SUFFIX] / filenames[self.MIX_SUFFIX],
            'target_wave': filedirs[self.TARGET_SUFFIX] / filenames[self.TARGET_SUFFIX],
            'ref_wave': filedirs[self.REF_SUFFIX] / filenames[self.REF_SUFFIX],
        }

    def get_index(self) -> Dict[str, Dict]:
        index = defaultdict(dict)  # mix_id: paths
        dirpaths = {suf: self._dirpath / dirname for suf, dirname in zip(self.SUFFIXES, self.DIRNAMES)}
        for suf, dirpath in dirpaths.items():
            for filename in os.listdir(dirpath):
                mix_id = filename.split('-')[0]
                assert os.path.splitext(filename)[0].endswith(f'-{suf}')
                index[mix_id][suf] = dirpath / filename

        final_index = {}
        for mix_id, filepaths in index.items():
            if len(filepaths) < 3:
                print(f'Some files are missing for mix \"{mix_id}\"')
            else:
                final_index[mix_id] = {
                    'mixed_wave': filepaths[self.MIX_SUFFIX],
                    'target_wave': filepaths[self.TARGET_SUFFIX],
                    'ref_wave': filepaths[self.REF_SUFFIX],
                }

        return final_index
