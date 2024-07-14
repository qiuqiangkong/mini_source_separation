import torch
import soundfile
import librosa

from data.musdb18hq import MUSDB18HQ
from data.audio_io import load


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


if __name__ == '__main__':

    # add()

    add2()