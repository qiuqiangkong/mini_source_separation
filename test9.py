import torch
import soundfile
import librosa
import numpy as np

from data.musdb18hq import MUSDB18HQ
import matplotlib.pyplot as plt
# from data.audio_io import load


def add():

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

    for i, source_type in enumerate(data.keys()):
        soundfile.write(file="_zz_{}.wav".format(source_type), data=data[source_type][0].data.cpu().numpy().T, samplerate=44100)


def add2():

    path = "/datasets/musdb18hq/train/Leaf - Wicked/mixture.wav"
    duration = librosa.get_duration(path=path)
    audio = load(path=path, sr=44100, mono=False, offset=duration, duration=2.)
    from IPython import embed; embed(using=False); os._exit(0)


def add3():

    a1 = torch.rsqrt(torch.Tensor([2]))

    from IPython import embed; embed(using=False); os._exit(0)


def add4():

    from models.rotary import RotaryEmbedding
    rotary_emb = RotaryEmbedding(dim = 32)

    # mock queries and keys - dimensions should end with (seq_len, feature dimension), and any number of preceding dimensions (batch, heads, etc)

    q = torch.randn(1, 8, 1024, 64) # queries - (batch, heads, seq len, dimension of head)
    k = torch.randn(1, 8, 1024, 64) # keys

    # apply the rotations to your queries and keys after the heads have been split out, but prior to the dot product and subsequent softmax (attention)

    q = rotary_emb.rotate_queries_or_keys(q)
    k = rotary_emb.rotate_queries_or_keys(k) 


def add5():

    import torch

    from models.rotary import (
        RotaryEmbedding,
        apply_rotary_emb
    )

    pos_emb = RotaryEmbedding(
        dim = 16,
        freqs_for = 'pixel',
        max_freq = 256
    )

    # queries and keys for frequencies to be rotated into
    # say for a video with 8 frames, and rectangular image (feature dimension comes last)

    # q = torch.randn(1, 8, 64, 32, 64)
    # k = torch.randn(1, 8, 64, 32, 64)
    q = torch.randn(1, 12, 8, 64, 32, 64)
    k = torch.randn(1, 12, 8, 64, 32, 64)

    # get axial frequencies - (8, 64, 32, 16 * 3 = 48)
    # will automatically do partial rotary

    freqs = pos_emb.get_axial_freqs(8, 64, 32)
    # (8, 64, 32, 48)

    # rotate in frequencies

    q = apply_rotary_emb(freqs, q)
    k = apply_rotary_emb(freqs, k)

    from IPython import embed; embed(using=False); os._exit(0)


def add6():

    sr = 44100
    n_fft = 2048
    hop_length = 441
    n_mels = 256

    audio_path = "assets/vocals_accompaniment_10s.wav"
    audio, fs = librosa.load(path=audio_path, sr=sr, mono=True)
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, window='hann', center=True).T
    # (T, F)

    melbanks = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels - 2, norm=None)
    # (256, 1025)

    melbank_first = np.zeros(melbanks.shape[-1])
    melbank_first[0] = 1.

    melbank_last = np.zeros(melbanks.shape[-1])
    idx = np.argmax(melbanks[-1])
    melbank_last[idx :] = 1. - melbanks[-1, idx :]
    
    melbanks = np.concatenate(
        [melbank_first[None, :], melbanks, melbank_last[None, :]], axis=0
    )

    sum_banks = np.sum(melbanks, axis=0)
    assert np.allclose(a=sum_banks, b=1.)

    
    # melbanks.

    melbanks = torch.Tensor(melbanks)
    stft = torch.Tensor(stft)

    indexes = []

    y = torch.zeros_like(stft)
    sum_banks = torch.zeros(stft.shape[1])

    for f in range(melbanks.shape[0]):
        idxes = (melbanks[f] != 0).nonzero().squeeze()
        indexes.append(idxes)

        bank = melbanks[f, idxes]  # (banks,)
        stft_bank = stft[:, idxes]  # (T, banks)
        y[:, idxes] += stft_bank * bank

    (stft - y).abs().mean()

    from IPython import embed; embed(using=False); os._exit(0)    

    plt.matshow(melbanks, origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz.pdf")

    # import librosa

    # audio, fs = librosa.load(path="123.wav", sr=24000, mono=True)

    # stft = librosa.stft(
    #     y=audio, 
    #     n_fft=1024, 
    #     hop_length=240, 
    #     window='hann', 
    #     center=True
    # )


    # from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':

    # add()

    add6()