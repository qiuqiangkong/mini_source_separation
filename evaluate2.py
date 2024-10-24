import argparse
from pathlib import Path
from typing import Union, Optional
import numpy as np
import os
from tqdm import tqdm
from data.audio import load
import librosa
import museval

import torch
from torch import nn

from data.musdb18hq import MUSDB18HQ
from train import get_model, separate


def evaluate(args):

    model_name = args.model_name
    ckpt_path = args.ckpt_path
    clip_duration = args.clip_duration
    batch_size = args.batch_size
    evaluate_num = None

    root = "/datasets/musdb18hq"
    split = "test"
    sr = 44100
    device = "cuda"
    source_types = MUSDB18HQ.source_types

    model = get_model(model_name)
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)

    sdr = validate2(
        root=root, 
        split=split, 
        sr=sr,
        clip_duration=clip_duration,
        source_types=source_types, 
        target_source_type="vocals",
        batch_size=batch_size,
        model=model,
        evaluate_num=evaluate_num,
        verbose=True
    )

    print("--- Median SDR ---")
    print("{:.2f} dB".format(sdr))


def validate2(
    root: str, 
    split: Union["train", "test"], 
    sr: int, 
    clip_duration: float, 
    source_types: list, 
    target_source_type: str, 
    batch_size: int, 
    model: nn.Module, 
    evaluate_num: Optional[int],
    verbose: bool = False
) -> float:
    r"""Calculate SDR.
    """

    clip_samples = round(clip_duration * sr)

    audios_dir = Path(root, split)
    audio_names = sorted(os.listdir(audios_dir))

    all_sdrs = []

    if evaluate_num:
        audio_names = audio_names[0 : evaluate_num]

    for audio_name in tqdm(audio_names):

        data = {}

        for source_type in source_types:
            audio_path = Path(audios_dir, audio_name, "{}.wav".format(source_type))

            audio = load(
                audio_path,
                sr=sr,
                mono=False
            )
            # shape: (channels, audio_samples)

            data[source_type] = audio

        data["mixture"] = np.sum([
            data[source_type] for source_type in source_types], axis=0)

        sep_wav = separate2(
            model=model, 
            audio=data["mixture"], 
            clip_samples=clip_samples,
            batch_size=batch_size
        )

        target_wav = data[target_source_type]

        # Calculate SDR. Shape should be (sources_num, channels_num, audio_samples)
        (sdrs, _, _, _) = museval.evaluate([target_wav.T], [sep_wav.T])

        sdr = np.nanmedian(sdrs)
        all_sdrs.append(sdr)

        if verbose:
            print(audio_name, "{:.2f} dB".format(sdr))

        # from IPython import embed; embed(using=False); os._exit(0)
        # import soundfile
        # soundfile.write(file="_zz.wav", data=sep_wav[0, :], samplerate=44100)
        # from IPython import embed; embed(using=False); os._exit(0)
        # soundfile.write(file="_zz1.wav", data=outputs[21, 0], samplerate=44100)

    sdr = np.nanmedian(all_sdrs)

    return sdr


def enframe(x, frame_length, hop_length):

    t = 0
    audio_samples = x.shape[-1]
    clips = []

    frames_num = int(np.ceil(audio_samples / hop_length))
    N = (frames_num - 1) * hop_length + frame_length
    divide_values = np.zeros(N)

    while t < audio_samples:
        
        clip = x[:, t : t + frame_length]
        clip = librosa.util.fix_length(data=clip, size=frame_length, axis=-1)
        clips.append(clip)

        divide_values[t : t + frame_length] += 1

        t += hop_length

    clips = np.stack(clips, axis=0)
    return clips, divide_values
    

def deframe(xs, divide_values, frame_length ,hop_length, audio_samples):

    N = (len(xs) - 1) * hop_length + frame_length
    sep_wav = np.zeros((2, N))

    for n in range(len(xs)):
        t = n * hop_length
        sep_wav[:, t : t + frame_length] += xs[n]

    sep_wav /= divide_values
    sep_wav = sep_wav[:, 0 : audio_samples]
    return sep_wav


def separate2(
    model: nn.Module, 
    audio: np.ndarray, 
    clip_samples: int, 
    batch_size: int
) -> np.ndarray:
    r"""Separate a long audio.
    """

    device = next(model.parameters()).device

    audio_samples = audio.shape[1]
    padded_audio_samples = round(np.ceil(audio_samples / clip_samples) * clip_samples)
    audio = librosa.util.fix_length(data=audio, size=padded_audio_samples, axis=-1)

    hop_length = 4410
    # hop_length = 44100  # 10.66 dB
    # hop_length = 132300  # 10.54 dB
    clips, divide_values = enframe(x=audio, frame_length=clip_samples, hop_length=hop_length)
    # shape: (clips_num, channels_num, clip_samples)

    clips_num = clips.shape[0]

    pointer = 0
    outputs = []

    while pointer < clips_num:

        batch_clips = torch.Tensor(clips[pointer : pointer + batch_size].copy()).to(device)

        with torch.no_grad():
            model.eval()
            batch_output = model(mixture=batch_clips)
            batch_output = batch_output.cpu().numpy()

        outputs.append(batch_output)
        pointer += batch_size

    outputs = np.concatenate(outputs, axis=0)

    sep_wav = deframe(
        xs=outputs, 
        divide_values=divide_values,
        frame_length=clip_samples, 
        hop_length=hop_length, 
        audio_samples=audio_samples
    )

    # shape: (channels_num, audio_samples)
    # from IPython import embed; embed(using=False); os._exit(0)
    # soundfile.write(file="_zz.wav", data=sep_wav[0, :], samplerate=44100)
    # soundfile.write(file="_zz1.wav", data=outputs[21, 0], samplerate=44100)

    return sep_wav


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="UNet")
    parser.add_argument('--ckpt_path', type=str, default="./train/UNet/latest.pth")
    parser.add_argument('--clip_duration', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    evaluate(args)