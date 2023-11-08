from abc import abstractmethod
from pathlib import Path
from typing import Dict, List


class AudiosPerSpeakerIndexer:
    @abstractmethod
    def get_index(self) -> Dict[str, List[Path]]:
        """
        :return: {speaker: [path1, ...]}
        """
        raise NotImplementedError()
