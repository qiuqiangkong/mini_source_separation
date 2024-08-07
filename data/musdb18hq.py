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

    duration = 35359.56  # Dataset duration (s), including training, valdiation, and testing

    source_types = ["bass", "drums", "other", "vocals"]
    acc_source_types = ["bass", "drums", "other"]

    def __init__(
        self,
        root: str = None, 
        split: ["train", "test"] = "train",
        sr: int = 44100,
        crop: Optional[callable] = RandomCrop(clip_duration=2.),
        remix_prob: float = 0.5,  # Remix different stems probability (between 0 and 1)
        transform: Optional[callable] = None,
    ):

        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.remix_prob = remix_prob
        self.transform = transform
        
        if not Path(self.root).exists():
            raise Exception("Please download the MUSDB18HQ dataset from {}".format(MUSDB18HQ.url))

        assert 0. <= self.remix_prob <= 1.

        self.audios_dir = Path(self.root, self.split)
        self.audio_names = sorted(os.listdir(self.audios_dir))
        
    def __getitem__(self, index: int) -> dict:

        source_types = MUSDB18HQ.source_types
        acc_source_types = MUSDB18HQ.acc_source_types

        audio_name = self.audio_names[index]
    
        # Sample a bool value indicating whether to remix or not
        remix = random.choices(
            population=(True, False), 
            weights=(self.remix_prob, 1. - self.remix_prob)
        )[0]

        # Get shared start time
        audio_path = Path(self.audios_dir, audio_name, "vocals.wav")
        audio_duration = librosa.get_duration(path=audio_path)
        shared_start_time, clip_duration = self.crop(audio_duration=audio_duration)

        data = {
            "dataset_name": "MUSDB18HQ",
            "audio_path": str(audio_path),
        }

        for source_type in source_types:

            audio_path = Path(self.audios_dir, audio_name, "{}.wav".format(source_type))
            
            if remix:
                # If remix, then each stem will has a different start time
                audio_duration = librosa.get_duration(path=audio_path)
                start_time, _ = self.crop(audio_duration=audio_duration)
            else:
                # If not remix, all stems will share the same start time
                start_time = shared_start_time

            # Load audio
            audio = load(
                path=audio_path, 
                sr=self.sr, 
                offset=start_time, 
                duration=clip_duration
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
    

if __name__ == "__main__":
    r"""Example.
    """
    
    from torch.utils.data import DataLoader

    root = "/datasets/musdb18hq"

    sr = 44100

    dataset = MUSDB18HQ(
        root=root,
        split="train",
        sr=sr,
        crop=RandomCrop(clip_duration=2., end_pad=0.),
    )

    dataloader = DataLoader(dataset=dataset, batch_size=4)

    for data in dataloader:

        n = 0
        audio_path = data["audio_path"][n]
        vocals_start_time = data["vocals_start_time"][n].cpu().numpy()
        bass_start_time = data["bass_start_time"][n].cpu().numpy()
        drums_start_time = data["drums_start_time"][n].cpu().numpy()
        other_start_time = data["other_start_time"][n].cpu().numpy()
        bass = data["bass"][n].cpu().numpy()
        drums = data["drums"][n].cpu().numpy()
        other = data["other"][n].cpu().numpy()
        vocals = data["vocals"][n].cpu().numpy()
        accompaniment = data["accompaniment"][n].cpu().numpy()
        mixture = data["mixture"][n].cpu().numpy()
        break

    # ------ Visualize ------
    print("audio_path:", audio_path)
    print("vocals_start_time:", vocals_start_time)
    print("bass_start_time:", bass_start_time)
    print("drums_start_time:", drums_start_time)
    print("other_start_time:", other_start_time)
    print("mixture:", mixture.shape)

    import soundfile
    soundfile.write(file="out_bass.wav", data=bass.T, samplerate=sr)
    soundfile.write(file="out_drums.wav", data=drums.T, samplerate=sr)
    soundfile.write(file="out_other.wav", data=other.T, samplerate=sr)
    soundfile.write(file="out_vocals.wav", data=vocals.T, samplerate=sr)
    soundfile.write(file="out_acc.wav", data=accompaniment.T, samplerate=sr)
    soundfile.write(file="out_mixture.wav", data=mixture.T, samplerate=sr)