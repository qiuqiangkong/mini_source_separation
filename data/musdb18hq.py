import torch
from pathlib import Path
import pandas as pd
import random
import os
import torchaudio
from pathlib import Path
import librosa


class Musdb18HQ:
	
	def __init__(self,
		root: str = None, 
		split: str = "train",
		segment_seconds: float = 2.0,
	):

		self.root = root
		self.split = split
		self.segment_seconds = segment_seconds

		self.audios_dir = Path(self.root, self.split)
		self.audio_names = sorted(os.listdir(self.audios_dir))
		self.audios_num = len(self.audio_names)
		
		self.source_types = ["mixture", "vocals"]

		self.check_exists()
		
	def __getitem__(self, index):

		# Sample an audio name from all audios.
		audio_index = random.randint(0, self.audios_num - 1)
		audio_name = self.audio_names[audio_index]

		data_dict = {}
		
		audio_path = Path(self.audios_dir, audio_name, "mixture.wav")
		duration = librosa.get_duration(path=audio_path)
		orig_sr = librosa.get_samplerate(path=audio_path)

		# Sample a short segment from a full audio.
		segment_start_time = random.uniform(0, duration - self.segment_seconds)
		segment_start_sample = int(segment_start_time * orig_sr)
		segment_samples = int(self.segment_seconds * orig_sr)

		for source_type in self.source_types:

			audio_path = Path(self.audios_dir, audio_name, "{}.wav".format(source_type))

			segment, _ = torchaudio.load(
				audio_path, 
				frame_offset=segment_start_sample, 
				num_frames=segment_samples
			)
			# shape: (channels, audio_samples)

			data_dict[source_type] = segment

		return data_dict

	def __len__(self):

		return 1000	 # Let each epoch contains 1000 data samples.


	def check_exists(self):

		if not Path(self.root).exists():
			raise Exception("MUSDB18HQ does not exist on the disk! Please download it from https://zenodo.org/records/3338373 and unzip it.")