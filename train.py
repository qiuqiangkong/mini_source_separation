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


def train(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    device = "cuda"
    epochs = 20
    checkpoints_dir = Path("./checkpoints", model_name)
    debug = False
    
    root = "/home/qiuqiangkong/datasets/musdb18hq"

    # Dataset
    dataset = Musdb18HQ(
        root=root,
        split="train",
        segment_seconds=2.,
    )

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=4, 
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
    for epoch in range(1, epochs):
        
        for data in tqdm(dataloader):

            mixture = data["mixture"].to(device)
            target = data["vocals"].to(device)

            # Play the audio.
            if debug:
                play_audio(mixture, target)

            optimizer.zero_grad()

            model.train()
            output = model(mixture=mixture) 

            loss = l1_loss(output, target)
            loss.backward()

            optimizer.step()

        print(loss)

        # Save model
        if epoch % 2 == 0:
            checkpoint_path = Path(checkpoints_dir, "epoch={}.pth".format(epoch))
            torch.save(model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

            checkpoint_path = Path(checkpoints_dir, "latest.pth")
            torch.save(model.state_dict(), Path(checkpoint_path))
            print("Save model to {}".format(checkpoint_path))


def get_model(model_name):
    if model_name == "UNet":
        return UNet()
    else:
        raise NotImplementedError


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