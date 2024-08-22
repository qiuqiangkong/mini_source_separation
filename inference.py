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

from train import get_model, separate


def inference(args):

    # Arguments
    model_name = args.model_name
    ckpt_path = args.ckpt_path
    clip_duration = args.clip_duration
    batch_size = args.batch_size

    # Default parameters
    sr = 44100
    clip_samples = round(clip_duration * sr)
    device = "cuda"

    # Load model
    model = get_model(model_name)
    model.load_state_dict(torch.load(ckpt_path))
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
    parser.add_argument('--ckpt_path', type=str, default="./train/UNet/latest.pth")
    parser.add_argument('--clip_duration', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    inference(args)