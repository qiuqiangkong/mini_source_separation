import torch
import time
import librosa
import numpy as np
import soundfile
from pathlib import Path
import torch.optim as optim
from data.musdb18hq import Musdb18HQ
from data.collate import collate_fn
from models.unet import UNet
from tqdm import tqdm
import museval
import argparse


def inference(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    segment_seconds = 2.
    device = "cuda"

    # Load checkpoint
    checkpoint_path = Path("checkpoints", model_name, "latest.pth")

    model = UNet()
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Load audio. Change this path to your favorite song.
    root = "/home/qiuqiangkong/datasets/musdb18hq/test"
    mixture_path = Path(root, "Al James - Schoolboy Facination", "mixture.wav") 

    mixture, orig_sr = librosa.load(path=mixture_path, sr=None, mono=False)
    # (channels_num, audio_samples)

    audio_samples = mixture.shape[-1]
    sep_wavs = []
    bgn = 0
    segment_samples = int(segment_seconds * orig_sr)

    # Do separation
    while bgn < audio_samples:
        
        print("Processing: {:.1f} s".format(bgn / orig_sr))

        # Cut segments
        segment = mixture[:, bgn : bgn + segment_samples]
        segment = librosa.util.fix_length(data=segment, size=segment_samples, axis=-1)
        segment = torch.Tensor(segment).to(device)

        # Separate a segment
        with torch.no_grad():
            model.eval()
            sep_wav = model(mixture=segment[None, :, :])[0]
            sep_wavs.append(sep_wav.cpu().numpy())

        bgn += segment_samples
            
    sep_wavs = np.concatenate(sep_wavs, axis=-1)
    sep_wavs = sep_wavs[:, 0 : audio_samples]

    # Write out separated audio
    sep_path = "sep.wav"
    soundfile.write(file=sep_path, data=sep_wavs.T, samplerate=orig_sr)
    print("Write to {}".format(sep_path))

    # Calculate SDR if there are ground truth (optional).
    target_path = Path(root, "Al James - Schoolboy Facination", "vocals.wav") 
    target, _ = librosa.load(path=target_path, sr=None, mono=False)

    (sdrs, _, _, _) = museval.evaluate(references=[target.T], estimates=[sep_wavs.T])
    print("SDR: {:.3f}".format(np.nanmedian(sdrs)))
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="UNet")
    args = parser.parse_args()

    inference(args)