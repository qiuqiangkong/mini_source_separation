import torch
import soundfile

from data.musdb18hq import MUSDB18HQ


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


if __name__ == '__main__':

    add()