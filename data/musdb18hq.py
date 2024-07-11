import torch
from pathlib import Path
import pandas as pd
import random
import os
import numpy as np
from typing import List
import torchaudio
from pathlib import Path
import librosa
from torch.utils.data import Dataset

from data.audio_io import load


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
		│	... 
		│	└── ...
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
	source_types = ["bass", "drums", "other", "vocals"]
	acc_source_types = ["bass", "drums", "other"]

	def __init__(self,
		root: str = None, 
		split: ["train", "test"] = "train",
		sr: int = 44100,
		mono: bool = False,
		segment_duration: float = 2.0,
		remix_prob: float = 1.  # A probability between 0 and 1
	):

		self.root = root
		self.split = split
		self.sr = sr
		self.mono = mono
		self.segment_duration = segment_duration
		self.remix_prob = remix_prob

		if not Path(self.root).exists():
			raise Exception("Please download the MUSDB18HQ dataset from {}".format(url))

		assert 0. <= self.remix_prob <= 1.

		self.audios_dir = Path(self.root, self.split)
		self.audio_names = sorted(os.listdir(self.audios_dir))
		
	def __getitem__(self, index):

		source_types = MUSDB18HQ.source_types
		acc_source_types = MUSDB18HQ.acc_source_types

		audio_name = self.audio_names[index]

		if self.split == "train":
			# Sample a bool value indicating whether to remix or not
			remix = random.choices(
				population=(True, False), 
				weights=(self.remix_prob, 1. - self.remix_prob)
			)[0]

			# Get shared start time
			audio_path = Path(self.audios_dir, audio_name, "vocals.wav")
			shared_start_time = self.random_start_time(audio_path)

		elif self.split == "test":
			remix = False
			shared_start_time = 60.

		data = {}

		for source_type in source_types:

			audio_path = Path(self.audios_dir, audio_name, "{}.wav".format(source_type))

			if remix:
				# If remix, then each stem will has a different start time
				seg_start_time = self.random_start_time(audio_path)
			else:
				# If not remix, all stems will share the same start time
				seg_start_time = shared_start_time

			segment = load(
				audio_path,
				sr=self.sr,
				mono=self.mono,
				offset=seg_start_time,
				duration=self.segment_duration
			)
			# shape: (channels, audio_samples)

			data[source_type] = segment


		data["accompaniment"] = np.sum([
			data[source_type] for source_type in acc_source_types], axis=0)
		# shape: (channels, audio_samples)

		data["mixture"] = np.sum(list(data.values()), axis=0)
		# shape: (channels, audio_samples)

		return data

	def __len__(self) -> int:

		audios_num = len(self.audio_names)

		return audios_num

	def random_start_time(self, path):
		duration = librosa.get_duration(path=path)
		seg_start_time = random.uniform(0, duration)
		return seg_start_time
		

if __name__ == "__main__":

    # Example
    root = "/datasets/musdb18hq"

    dataset = MUSDB18HQ(
        root=root,
        split="train",
        sr=44100,
        mono=False,
        segment_duration=2.,
    )

    dataloader = torch.utils.data.DataLoader(dataset=dataset)

    for data in dataloader:
        print(data)
        break