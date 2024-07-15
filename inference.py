import torch
import time
import librosa
import numpy as np
import soundfile
from pathlib import Path
from models.unet import UNet
from tqdm import tqdm
import museval
import argparse

from train import separate


def inference(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    sr = 44100
    segment_seconds = 2.
    clip_samples = round(segment_seconds * sr)
    batch_size = 16
    device = "cuda"

    # Load checkpoint
    checkpoint_path = Path("checkpoints", "train", model_name, "latest.pth")

    model = UNet()
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Load audio. Change this path to your favorite song.
    root = "/datasets/musdb18hq/test"
    mixture_path = Path(root, "Al James - Schoolboy Facination", "mixture.wav") 

    mixture, orig_sr = librosa.load(path=mixture_path, sr=None, mono=False)
    # (channels_num, audio_samples)

    sep_wav = separate(
        model=model, 
        audio=mixture, 
        clip_samples=clip_samples, 
        batch_size=batch_size
    )

    # Write out separated audio
    sep_path = "sep.wav"
    soundfile.write(file=sep_path, data=sep_wav.T, samplerate=orig_sr)
    print("Write to {}".format(sep_path))
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="UNet")
    args = parser.parse_args()

    inference(args)