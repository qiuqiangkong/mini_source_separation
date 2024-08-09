import os
import random
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from torch.utils.data import Dataset

from data.audio import load
from data.crops import RandomCrop


class MUSDB18HQ(Dataset):
    r"""MUSDB18HQ [1] is a dataset containing 100 training audios and 50 
    testing audios with vocals, bass, drums, other stems. Audios are stereo and 
    sampled at 48,000 Hz. Dataset size is 30 GB.

    [1] https://zenodo.org/records/3338373

    The dataset looks like:

        dataset_root (30 GB)
        ├── train (100 files)
        │   ├── A Classic Education - NightOwl
        │   │   ├── bass.wav
        │   │   ├── drums.wav
        │   │   ├── mixture.wav
        │   │   ├── other.wav
        │   │   └── vocals.wav
        │   ... 
        │   └── ...
        └── test (50 files)
            ├── Al James - Schoolboy Facination
            │   ├── bass.wav
            │   ├── drums.wav
            │   ├── mixture.wav
            │   ├── other.wav
            │   └── vocals.wav
            ... 
            └── ...
    """

    url = "https://zenodo.org/records/3338373"

    duration = 35359.56  # Dataset duration (s), 9.8 hours, including training, 
    # valdiation, and testing

    source_types = ["bass", "drums", "other", "vocals"]
    acc_source_types = ["bass", "drums", "other"]

    def __init__(
        self,
        root: str = None, 
        split: ["train", "test"] = "train",
        sr: int = 44100,
        crop: callable = RandomCrop(clip_duration=2.),
        target_source_type: Optional[str] = "vocals",
        remix: dict = {"no_remix": 0.1, "half_remix": 0.4, "full_remix": 0.5},
        transform: Optional[callable] = None,
    ):

        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.target_source_type = target_source_type
        self.remix_types = list(remix.keys())
        self.remix_weights = list(remix.values())
        self.transform = transform

        assert np.sum(self.remix_weights) == 1.0
        
        if not Path(self.root).exists():
            raise Exception("Please download the MUSDB18HQ dataset from {}".format(MUSDB18HQ.url))

        self.audios_dir = Path(self.root, self.split)
        self.audio_names = sorted(os.listdir(self.audios_dir))
        
    def __getitem__(self, index: int) -> dict:

        source_types = MUSDB18HQ.source_types
        acc_source_types = MUSDB18HQ.acc_source_types

        audio_name = self.audio_names[index]

        data = {
            "dataset_name": "MUSDB18HQ",
            "audio_path": str(Path(self.audios_dir, audio_name)),
        }

        remix_type = random.choices(
            population=self.remix_types,
            weights=self.remix_weights
        )[0]

        audio_path = Path(self.audios_dir, audio_name, "mixture.wav")
        audio_duration = librosa.get_duration(path=audio_path)

        start_time_dict = self.get_start_times(
            audio_duration=audio_duration, 
            source_types=source_types, 
            target_source_type=self.target_source_type,
            remix_type=remix_type
        )

        for source_type in source_types:

            audio_path = Path(self.audios_dir, audio_name, "{}.wav".format(source_type))
            start_time = start_time_dict[source_type]

            # Load audio
            audio = load(
                path=audio_path, 
                sr=self.sr, 
                offset=start_time, 
                duration=self.crop.clip_duration
            )
            # shape: (channels, audio_samples)

            if self.transform is not None:
                audio = self.transform(audio)

            data[source_type] = audio
            data["{}_start_time".format(source_type)] = start_time

        data["accompaniment"] = np.sum([
            data[source_type] for source_type in acc_source_types], axis=0)
        # shape: (channels, audio_samples)

        data["mixture"] = np.sum([
            data[source_type] for source_type in source_types], axis=0)
        # shape: (channels, audio_samples)

        return data

    def __len__(self) -> int:

        audios_num = len(self.audio_names)

        return audios_num

    def get_start_times(
        self, 
        audio_duration: float, 
        source_types: str, 
        target_source_type: str,
        remix_type: str
    ) -> dict:

        start_time_dict = {}

        if remix_type == "no_remix":
            
            start_time1, _ = self.crop(audio_duration=audio_duration)

            for source_type in source_types:
                start_time_dict[source_type] = start_time1

        elif remix_type == "half_remix":
            
            start_time1, _ = self.crop(audio_duration=audio_duration)
            start_time2, _ = self.crop(audio_duration=audio_duration)

            for source_type in source_types:
                if source_type == target_source_type:
                    start_time_dict[source_type] = start_time1
                else:
                    start_time_dict[source_type] = start_time2

        elif remix_type == "full_remix":

            for source_type in source_types:
                start_time, _ = self.crop(audio_duration=audio_duration)
                start_time_dict[source_type] = start_time

        else:
            raise NotImplementedError

        return start_time_dict