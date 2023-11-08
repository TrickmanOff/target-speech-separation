import json

import tss_lib.mixtures_generator.audios_per_speaker_datasets as audios_per_speaker_datasets_module
from tss_lib.mixtures_generator.audios_per_speaker_datasets import AudiosPerSpeakerIndexer
from tss_lib.config_processing.parse_config import ConfigParser


if __name__ == '__main__':
    config_dict = json.load(open('configs/mixtures_generator/librispeech-val.json', 'r'))
    config = ConfigParser(config_dict)
    dataset: AudiosPerSpeakerIndexer = config.init_obj(config["data"]["dataset"], audios_per_speaker_datasets_module,
                                                       config_parser=config)
    print(dataset.get_index())
