"""
script providing CLI interface for mixtures generation
"""
import argparse
import json
from pathlib import Path

import tss_lib.mixtures_generator.audios_per_speaker_datasets as audios_per_speaker_datasets_module
import tss_lib.mixtures_generator as generators_module
from tss_lib.config_processing.parse_config import ConfigParser
from tss_lib.datasets.mixtures_storage import SimpleMixturesStorage
from tss_lib.mixtures_generator.audios_per_speaker_datasets import AudiosPerSpeakerIndexer
from tss_lib.mixtures_generator.generator import MixtureGenerator, SpeakerFiles


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Mixtures Generator")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="path to the directory for mixtures to be stored",
    )
    args = args.parse_args()

    config_path = Path(args.config)
    out_dirpath = Path(args.output)

    # ----

    config_dict = json.load(open(config_path, 'r'))
    config = ConfigParser(config_dict)
    dataset: AudiosPerSpeakerIndexer = config.init_obj(config["data"]["dataset"],
                                                       audios_per_speaker_datasets_module,
                                                       config_parser=config)
    speakers_index = dataset.get_index()
    speakers_files = [
        SpeakerFiles(id, files) for id, files in speakers_index.items()
    ]

    mix_storage = SimpleMixturesStorage(out_dirpath)
    generator: MixtureGenerator = config.init_obj(config["mixtures_generator"]["generator"],
                                                  generators_module,
                                                  speakers_files=speakers_files,
                                                  out_storage=mix_storage)

    generator.generate_mixes(**config["mixtures_generator"]["params"])
