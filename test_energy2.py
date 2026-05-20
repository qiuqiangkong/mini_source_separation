from pathlib import Path
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch import Tensor
import torch.nn as nn
import math
from scipy.signal import firwin
import torch.nn.functional as F
from einops import rearrange
import soundfile
import time
import h5py

from test_energy2_band import SubbandFilter
from mss.utils import fast_sdr




def add():

    audio_path = "assets/music_10s.wav"
    sr = 48000
    device = "cuda"

    banks = [[0, 4000], [4000, 24000]]
    sb_filter = SubbandFilter(sr, banks)

    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
    audio = Tensor(audio)[None, None, :]
    y = sb_filter.analysis(audio)  # (b, c, k, l)
    z = sb_filter.synthesis(y)

    soundfile.write(file="_zz0.wav", data=y[0, 0, 0].data.cpu().numpy(), samplerate=sr)
    soundfile.write(file="_zz1.wav", data=y[0, 0, 1].data.cpu().numpy(), samplerate=sr)
    soundfile.write(file="_zz2.wav", data=z[0, 0].data.cpu().numpy(), samplerate=sr)

    sdr = fast_sdr(audio.cpu().numpy(), z.cpu().numpy())
    print(f"SDR: {sdr:02f} dB")
    from IPython import embed; embed(using=False); os._exit(0)


def add2():

    audio_path = "assets/music_10s.wav"
    sr = 48000
    device = "cuda"
    segment_samples = int(2 * sr)

    root = "datasets/musdb18hq/train"
    paths = list(Path(root).rglob('mixture.wav'))


    # banks = [[0, 4000], [4000, 24000]]
    # freqs = [0, 4000, 24000]
    f_low = 0
    f_mid = 4000
    # f_mid = 10
    f_high = 24000
    # banks = [[f_low, f_mid], [f_mid, f_high]]
    # sb_filter = SubbandFilter(sr, banks).to(device)

    op = op_energy

    while f_high - f_low > 1:
        print(f_low, f_mid, f_high)

        banks = [[f_low, f_mid], [f_mid, f_high]]
        sb_filter = SubbandFilter(sr, banks).to(device)

        value_low = []
        value_high = []

        for n, path in enumerate(paths):
            # print(n)
            audio, _ = librosa.load(path=path, sr=sr, mono=True)
            
            y = forward_in_chunks(sb_filter, audio, segment_samples, device)

            for i in range(len(y)):
                y_low = y[i, 0, 0, :]
                y_high = y[i, 0, 1, :]

                value_low.append(op(y_low).item())
                value_high.append(op(y_high).item())

            if n == 2:
                break

        value_low = np.mean(value_low)
        value_high = np.mean(value_high)

        if value_low > value_high:
            f_high = f_mid
        else:
            f_low = f_mid
        
        f_mid = (f_low + f_high) / 2



    from IPython import embed; embed(using=False); os._exit(0)

    # sdr = fast_sdr(audio.cpu().numpy(), z.cpu().numpy())
    # print(f"SDR: {sdr:02f} dB")
    # from IPython import embed; embed(using=False); os._exit(0)


def forward_in_chunks(sb_filter: nn.Module, x: np.ndarray, segment_samples: int, device) -> Tensor:

    x = librosa.util.frame(x, frame_length=segment_samples, hop_length=segment_samples).T  # (n, l)
    x = Tensor(x.copy())[:, None, :].to(device)
    y = sb_filter.analysis(x)  # (b, c, k, l)

    # y = rearrange(y, 'b 1 k l -> k (b l)')
    return y


def op_energy(x):
    return (x**2).mean()


def op_abs(x):
    return x.abs().mean()


def add3():

    out = sub(0, 24000, depth=0)
    print("================")
    print(out)


def sub(low, high, depth):

    mid = mul(low, high)
    
    if depth == 5:
        return [(low, high)]
    
    out = []    
    out += sub(low, mid, depth + 1)
    out += sub(mid, high, depth + 1)
    return out

    
    

def mul(low, high):
    return (low + high) / 2


def add4():

    t1 = time.time()
    searcher = Searcher()

    # out = dfs(searcher, low=0, high=24000, depth=0, max_depth=3)
    out = dfs(searcher, low=0, high=24000, depth=0, max_depth=7)
    print("================")
    print(out)

    out_path = "_zz_abs.csv" 
    with open(out_path, 'w') as fw:
        for i in range(len(out)):
            fw.write(str(out[i][0]) + '\n')
        fw.write(str(out[i][1]) + '\n')
        
    print(f"Write out to {out_path}")
    print("Time: {:.2f} s".format(time.time() - t1))
    from IPython import embed; embed(using=False); os._exit(0)


def add5():

    sr = 48000
    root = "datasets/musdb18hq/train"
    paths = list(Path(root).rglob('mixture.wav'))
    out_dir = "_musdb18hq_hdf5"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for n, path in enumerate(paths):
        audio, _ = librosa.load(path=path, sr=sr, mono=True)
        out_path = Path(out_dir, f"{n:04d}.h5")
        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset("x", data=audio, dtype=np.float32)
        print(f"Write out to {out_path}")
        


def dfs(searcher, low, high, depth, max_depth):
    
    mid = searcher(low, high)
    # mid = mul(low, high)
    print("mid:", mid)
    
    if depth == max_depth:
        return [(low, high)]
    
    out = []    
    out += dfs(searcher, low, mid, depth + 1, max_depth)
    out += dfs(searcher, mid, high, depth + 1, max_depth)
    
    return out


'''
class Searcher:
    def __init__(self):
        
        root = "datasets/musdb18hq/train"
        self.paths = list(Path(root).rglob('mixture.wav'))

        self.sr = 48000
        self.segment_samples = int(2 * self.sr)
        self.device = "cuda"
        self.op = op_abs

    def __call__(self, f_low, f_high):

        f_low0 = f_low
        f_high0 = f_high
        f_mid = (f_low + f_high) / 2

        while True:

            banks = [[f_low0, f_mid], [f_mid, f_high0]]
            print(banks)
            sb_filter = SubbandFilter(self.sr, banks).to(self.device)

            value_low = []
            value_high = []

            for n, path in enumerate(self.paths):

                audio, _ = librosa.load(path=path, sr=self.sr, mono=True)
                
                y = forward_in_chunks(sb_filter, audio, self.segment_samples, self.device)

                for i in range(len(y)):
                    y_low = y[i, 0, 0, :]
                    y_high = y[i, 0, 1, :]

                    value_low.append(self.op(y_low).item())
                    value_high.append(self.op(y_high).item())

                # if n == 2:
                #     break

            value_low = np.mean(value_low)
            value_high = np.mean(value_high)
            # print("^", value_low, value_high)

            if value_low > value_high:
                f_mid_new = (f_low + f_mid) / 2
                f_high = f_mid
            else:
                f_mid_new = (f_mid + f_high) / 2
                f_low = f_mid

            if np.abs(f_mid - f_mid_new) < 1.:
                break
            else:
                f_mid = f_mid_new

        return f_mid
'''

class Searcher:
    def __init__(self):
        
        # root = "datasets/musdb18hq/train"
        root = "_musdb18hq_hdf5"
        self.paths = list(Path(root).rglob('*.h5'))

        self.sr = 48000
        self.segment_samples = int(2 * self.sr)
        self.device = "cuda"
        self.op = op_abs

    def __call__(self, f_low, f_high):

        f_low0 = f_low
        f_high0 = f_high
        f_mid = (f_low + f_high) / 2

        while True:

            banks = [[f_low0, f_mid], [f_mid, f_high0]]
            print(banks)
            sb_filter = SubbandFilter(self.sr, banks).to(self.device)

            value_low = []
            value_high = []

            for n, path in enumerate(self.paths):
                
                with h5py.File(path, 'r') as hf:
                    audio = hf["x"][:]
                
                y = forward_in_chunks(sb_filter, audio, self.segment_samples, self.device)

                for i in range(len(y)):
                    y_low = y[i, 0, 0, :]
                    y_high = y[i, 0, 1, :]

                    value_low.append(self.op(y_low).item())
                    value_high.append(self.op(y_high).item())

                # if n == 2:
                #     break

            value_low = np.mean(value_low)
            value_high = np.mean(value_high)
            # print("^", value_low, value_high)

            if value_low > value_high:
                f_mid_new = (f_low + f_mid) / 2
                f_high = f_mid
            else:
                f_mid_new = (f_mid + f_high) / 2
                f_low = f_mid

            if np.abs(f_mid - f_mid_new) < 1.:
                break
            else:
                f_mid = f_mid_new

        return f_mid


if __name__ == '__main__':

    add4()