import random
from typing import Optional

import librosa
import numpy as np
import torch
import torchaudio


def load(
    path: str, 
    sr: int, 
    offset: float = 0.,  # Load start time (s)
    duration: Optional[float] = None,  # Load duration (s)
    mono: bool = False
) -> np.ndarray:
    r"""Load audio.

    Returns:
       audio: (channels, audio_samples) 

    Examples:
        >>> audio = load_audio(path="xx/yy.wav", sr=16000)
    """
    
    # Prepare arguments
    orig_sr = librosa.get_samplerate(path)

    start_sample = round(offset * orig_sr)

    if duration:
        samples = round(duration * orig_sr)
    else:
        samples = -1

    # Load audio
    audio, fs = torchaudio.load(
        path, 
        frame_offset=start_sample, 
        num_frames=samples
    )
    # (channels, audio_samples)

    # Resample. Faster than librosa
    audio = torchaudio.functional.resample(
        waveform=audio, 
        orig_freq=orig_sr, 
        new_freq=sr
    ).numpy()
    # shape: (channels, audio_samples)

    if duration:
        new_samples = round(duration * sr)
        audio = librosa.util.fix_length(data=audio, size=new_samples, axis=-1)

    if mono:
        audio = np.mean(audio, axis=0, keepdims=True)
    
    return audio