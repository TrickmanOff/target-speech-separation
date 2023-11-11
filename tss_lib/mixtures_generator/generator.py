import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from tqdm import tqdm

from tss_lib.datasets.mixtures_storage import MixtureMeta, MixturesStorage


# copied from https://github.com/XuMuK1/dla2023/blob/2023/week05/semSpSep.ipynb


def snr_mixer(clean: np.ndarray, noise: np.ndarray, snr: float) -> np.ndarray:
    """
    returns a mixed_wave of form (clean + alpha*noise) such that
    SNR(clean, mixed_wave) = snr
    """
    amp_noise = np.linalg.norm(clean) / 10**(snr / 20)

    noise_norm = (noise / np.linalg.norm(noise)) * amp_noise

    mix = clean + noise_norm

    return mix


def vad_merge(w: np.ndarray, top_db: float) -> np.ndarray:
    """
    removes intervals of silence (with db < top_db)
    """
    intervals = librosa.effects.split(w, top_db=top_db)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)


def cut_audios(*ss: Union[np.ndarray, torch.Tensor], sec: float, sr: int,
               same_intervals: bool = True) -> Tuple[List[Union[np.ndarray, torch.Tensor]], ...]:
    """
    splits the audios `s1`, `s2` into intervals of `sec` seconds each
    assuming that both audios are parallel

    audios may be of shape (..., T)
    """
    cut_len = int(sr * sec)
    min_len = min(s.shape[-1] for s in ss)

    ss_cuts = [[] for _ in range(len(ss))]  # [[s1: ...], [s2: ...], ...]

    segment = 0
    while (segment + 1) * cut_len < min_len:
        for s, s_cuts in zip(ss, ss_cuts):
            s_cuts.append(s[..., segment * cut_len:(segment + 1) * cut_len])
        segment += 1
    if not same_intervals:
        assert all(s.shape[-1] == min_len for s in ss)
        for s, s_cuts in zip(ss, ss_cuts):
            s_cuts.append(s[..., segment * cut_len:])

    return tuple(ss_cuts)


def fix_length(s1: np.ndarray, s2: np.ndarray, min_or_max: str = 'max') -> Tuple[np.ndarray, np.ndarray]:
    """
    if `min_or_max` == 'min', then the longer audio is cut to be the same length as the other one
    if `min_or_max` == 'max', then the shorter audio is padded by zeros to be the same length as the other one
    """
    # Fix length
    if min_or_max == 'min':
        utt_len = np.minimum(len(s1), len(s2))
        s1 = s1[:utt_len]
        s2 = s2[:utt_len]
    elif min_or_max == 'max':  # max
        utt_len = np.maximum(len(s1), len(s2))
        s1 = np.append(s1, np.zeros(utt_len - len(s1)))
        s2 = np.append(s2, np.zeros(utt_len - len(s2)))
    else:
        raise RuntimeError()
    return s1, s2


def normalize_wave_loudness(wave: np.ndarray, target_loudness: float, sample_rate: int) -> np.ndarray:
    meter = pyln.Meter(sample_rate)  # create BS.1770 meter

    louds = meter.integrated_loudness(wave)
    wave_norm = pyln.normalize.loudness(wave, louds, target_loudness)
    return wave_norm


def create_mix(idx: int, triplet: Dict, snr_levels, out_storage: MixturesStorage,
               audio_len: Optional[float] = None,
               vad_db: Optional[float] = None, trim_db: Optional[float] = None,
               test: bool = False, sr: int = 16_000) -> None:
    """
    :param snr_levels: from these values the targets mixed_wave snr is chosen randomly
    :param idx: numerical index of the mixed_wave
    :param vad_db: a threshold of silence for mixing two audios
    :param audio_len: audios length in seconds for mixed_wave
    :param trim_db: leading and trailing sounds of all audios will be trimmed if they are
        quiter than this threshold
    """
    s1_path = triplet["targets"]
    s2_path = triplet["noise"]
    ref_path = triplet["reference"]
    target_id = triplet["target_id"]
    noise_id = triplet["noise_id"]

    s1, _ = sf.read(s1_path)
    s2, _ = sf.read(s2_path)
    ref, _ = sf.read(ref_path)
    meter = pyln.Meter(sr)  # create BS.1770 meter

    louds1 = meter.integrated_loudness(s1)
    louds2 = meter.integrated_loudness(s2)
    louds_ref = meter.integrated_loudness(ref)

    s1_norm = pyln.normalize.loudness(s1, louds1, -29)
    s2_norm = pyln.normalize.loudness(s2, louds2, -29)
    ref_norm = pyln.normalize.loudness(ref, louds_ref, -23.0)

    amp_s1 = np.max(np.abs(s1_norm))
    amp_s2 = np.max(np.abs(s2_norm))
    amp_ref = np.max(np.abs(ref_norm))

    if amp_s1 == 0 or amp_s2 == 0 or amp_ref == 0:
        return

    if trim_db:
        assert not test, 'Do not trim silence on test data'
        ref, _ = librosa.effects.trim(ref_norm, top_db=trim_db)
        s1, _ = librosa.effects.trim(s1_norm, top_db=trim_db)
        s2, _ = librosa.effects.trim(s2_norm, top_db=trim_db)

    if len(ref) < sr:
        return

    mix_id = f"{target_id}_{noise_id}_{idx:06d}"

    snr = np.random.choice(snr_levels, 1).item()

    if not test:
        assert audio_len is not None
        if vad_db is not None:
            s1, s2 = vad_merge(s1, vad_db), vad_merge(s2, vad_db)
        s1_cut, s2_cut = cut_audios(s1, s2, audio_len=audio_len, sr=sr)

        for i in range(len(s1_cut)):
            mix = snr_mixer(s1_cut[i], s2_cut[i], snr)

            louds1 = meter.integrated_loudness(s1_cut[i])
            s1_cut[i] = pyln.normalize.loudness(s1_cut[i], louds1, -23.0)
            loud_mix = meter.integrated_loudness(mix)
            mix = pyln.normalize.loudness(mix, loud_mix, -23.0)

            cur_mix_id = mix_id + f'_{i}'
            cur_out_filepaths = out_storage.get_mix_filepaths(cur_mix_id, with_ext='.wav')

            sf.write(cur_out_filepaths['mixed_wave'], mix, sr)
            sf.write(cur_out_filepaths['target_wave'], s1_cut[i], sr)
            sf.write(cur_out_filepaths['ref_wave'], ref, sr)
            out_storage.add_mix_meta(cur_mix_id, MixtureMeta(target_id, noise_id))
    else:
        out_filepaths = out_storage.get_mix_filepaths(mix_id, with_ext='.wav')

        s1, s2 = fix_length(s1, s2, 'max')
        mix = snr_mixer(s1, s2, snr)
        louds1 = meter.integrated_loudness(s1)
        s1 = pyln.normalize.loudness(s1, louds1, -23.0)

        loud_mix = meter.integrated_loudness(mix)
        mix = pyln.normalize.loudness(mix, loud_mix, -23.0)

        sf.write(out_filepaths['mixed_wave'], mix, sr)
        sf.write(out_filepaths['target_wave'], s1, sr)
        sf.write(out_filepaths['ref_wave'], ref, sr)
        out_storage.add_mix_meta(mix_id, MixtureMeta(target_id, noise_id))


@dataclass
class SpeakerFiles:
    id: str
    files: List[Path]


class MixtureGenerator:
    def __init__(self,
                 speakers_files: List[SpeakerFiles],
                 out_storage: MixturesStorage,
                 ntriplets: int = 5000,
                 test: bool = False,
                 random_state: int = 42):
        """
        :param ntriplets: How many triplets to generate
        """
        self.speakers_files = speakers_files  # list of SpeakerFiles for every speaker_id
        self.out_storage = out_storage
        self.ntriplets = ntriplets
        self.test = test
        self.random_state = random_state
        random.seed(self.random_state)

    def generate_triplets(self) -> Dict[str, List]:
        """
        generate triplets of files (ref_wave, targets, noise)
        each triplet is generated by randomly choosing speakers and their files
        """
        i = 0
        all_triplets = {"reference": [], "targets": [], "noise": [], "target_id": [], "noise_id": []}
        for i in tqdm(range(self.ntriplets), "Generating triplets"):
            spk1, spk2 = random.sample(self.speakers_files, 2)

            if len(spk1.files) < 2 or len(spk2.files) < 2:
                continue

            target, reference = random.sample(spk1.files, 2)
            noise = random.choice(spk2.files)
            all_triplets["reference"].append(reference)
            all_triplets["targets"].append(target)
            all_triplets["noise"].append(noise)
            all_triplets["target_id"].append(spk1.id)
            all_triplets["noise_id"].append(spk2.id)
            i += 1

        return all_triplets

    @staticmethod
    def triplet_generator(target_speaker: SpeakerFiles,
                          noise_speaker: SpeakerFiles,
                          number_of_triplets: int):
        """
        generate `number_of_triplets` triplets for particular targets and noise speakers
        """
        max_num_triplets = min(len(target_speaker.files), len(noise_speaker.files))
        number_of_triplets = min(max_num_triplets, number_of_triplets)

        target_samples = random.sample(target_speaker.files, k=number_of_triplets)
        reference_samples = random.sample(target_speaker.files, k=number_of_triplets)
        noise_samples = random.sample(noise_speaker.files, k=number_of_triplets)

        triplets = {"reference": [], "targets": [], "noise": [],
                    "target_id": [target_speaker.id] * number_of_triplets, "noise_id": [noise_speaker.id] * number_of_triplets}
        triplets["targets"] += target_samples
        triplets["reference"] += reference_samples
        triplets["noise"] += noise_samples

        return triplets

    def generate_mixes(self, snr_levels=(0,), num_workers=10, update_steps=10, **kwargs) -> None:
        """
        generate mixes from all triplets
        """
        triplets = self.generate_triplets()

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = []

            for i in range(self.ntriplets):
                triplet = {
                    "reference": triplets["reference"][i],
                    "targets": triplets["targets"][i],
                    "noise": triplets["noise"][i],
                    "target_id": triplets["target_id"][i],
                    "noise_id": triplets["noise_id"][i],
                }

                futures.append(pool.submit(create_mix, i, triplet,
                                           snr_levels, self.out_storage,
                                           test=self.test, **kwargs))

            for i, future in enumerate(futures):
                future.result()
                if (i + 1) % max(self.ntriplets // update_steps, 1) == 0:
                    print(f"Files Processed | {i + 1} out of {self.ntriplets}")
