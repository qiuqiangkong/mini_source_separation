import torch
import time
import librosa
import random
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


def train(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    device = "cuda"
    batch_size_per_device = 8
    num_workers = 8
    save_step_frequency = 2000
    training_steps = 100000
    debug = False
    devices_num = torch.cuda.device_count()

    print("Devices num: {}".format(devices_num))

    checkpoints_dir = Path("./checkpoints", model_name)
    
    # root = "./datasets/mini_musdb18hq"
    root = "/datasets/unzipped_packages/musdb18hq"

    # Dataset
    dataset = Musdb18HQ(
        root=root,
        split="train",
        segment_seconds=2.,
    )

    # Sampler
    sampler = Sampler(dataset_size=len(dataset))

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=8, 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=8, 
        pin_memory=True
    )

    # Model
    model = get_model(model_name)
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for step, data in enumerate(tqdm(dataloader)):

        mixture = data["mixture"].to(device)
        target = data["vocals"].to(device)

        # Play the audio.
        if debug:
            play_audio(mixture, target)

        optimizer.zero_grad()

        # Forward
        model.train()
        output = model(mixture=mixture) 

        # Backward
        loss = l1_loss(output, target)
        loss.backward()

        # Optimize
        optimizer.step()

        if step % 100 == 0:
            print("step: {}, loss: {:.3f}".format(step, loss.item()))

        # Save model
        if step % save_step_frequency == 0:
            checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

            checkpoint_path = Path(checkpoints_dir, "latest.pth")
            torch.save(model.state_dict(), Path(checkpoint_path))
            print("Save model to {}".format(checkpoint_path))

        if step == training_steps:
            break


def get_model(model_name):
    if model_name == "UNet":
        return UNet()
    else:
        raise NotImplementedError


class Sampler:
    def __init__(self, dataset_size):
        self.indexes = list(range(dataset_size))
        
    def __iter__(self):

        pointer = 0

        while True:

            if pointer == len(self.indexes):
                pointer = 0
                random.shuffle(self.indexes)

            index = self.indexes[pointer]
            pointer += 1

            yield index


def l1_loss(output, target):
    return torch.mean(torch.abs(output - target))


def play_audio(mixture, target):
    soundfile.write(file="tmp_mixture.wav", data=mixture[0].cpu().numpy().T, samplerate=44100)
    soundfile.write(file="tmp_target.wav", data=target[0].cpu().numpy().T, samplerate=44100)
    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="UNet")
    args = parser.parse_args()

    train(args)