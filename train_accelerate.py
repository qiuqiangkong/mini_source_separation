import argparse
import os
import random
from pathlib import Path
from typing import Union

import librosa
import museval
import numpy as np
from accelerate import Accelerator
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

wandb.require("core")

from data.audio import load
from data.musdb18hq import MUSDB18HQ
from data.crops import RandomCrop
from train import InfiniteSampler, get_model, l1_loss, validate, warmup_lambda


def train(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    sr = 44100
    mono = False
    clip_duration = 2.
    remix_prob = 0.8
    batch_size = 16
    num_workers = 16
    pin_memory = True
    learning_rate = 1e-3
    use_scheduler = True
    test_step_frequency = 5000
    save_step_frequency = 5000
    evaluate_num = 10
    training_steps = 100000
    wandb_log = True

    filename = Path(__file__).stem
    source_types = MUSDB18HQ.source_types

    checkpoints_dir = Path("./checkpoints", filename, model_name)
    
    root = "/datasets/musdb18hq"

    if wandb_log:
        wandb.init(project="mini_source_separation")

    # Training dataset
    train_dataset = MUSDB18HQ(
        root=root,
        split="train",
        sr=sr,
        crop=RandomCrop(clip_duration=clip_duration, end_pad=0.),
        remix_prob=remix_prob
    )

    # Samplers
    train_sampler = InfiniteSampler(train_dataset)

    # Dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    # Model
    model = get_model(model_name)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    if use_scheduler:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warmup_lambda)

    # Prepare for multiprocessing
    accelerator = Accelerator()

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler)

    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        mixture = data["mixture"]
        target = data["vocals"]

        # Forward
        model.train()
        output = model(mixture=mixture) 
        
        # Calculate loss
        loss = l1_loss(output, target)

        # Optimize
        optimizer.zero_grad()   # Reset all parameter.grad to 0
        accelerator.backward(loss)     # Update all parameter.grad
        optimizer.step()    # Update all parameters based on all parameter.grad

        # Learning rate scheduler (optional)
        if use_scheduler:
            scheduler.step()

        # Evaluate
        if step % test_step_frequency == 0:

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:

                if accelerator.num_processes == 1:
                    val_model = model
                else:
                    val_model = model.module
                
                sdrs = {}

                for split in ["train", "test"]:
            
                    sdr = validate(
                        root=root, 
                        split=split, 
                        sr=sr,
                        clip_duration=clip_duration,
                        source_types=source_types, 
                        target_source_type="vocals",
                        batch_size=batch_size,
                        model=val_model,
                        evaluate_num=evaluate_num,
                    )
                    sdrs[split] = sdr

                print("--- step: {} ---".format(step))
                print("Loss: {:.3f}".format(loss))
                print("Train SDR: {:.3f}".format(sdrs["train"]))
                print("Test SDR: {:.3f}".format(sdrs["test"]))

                if wandb_log:
                    wandb.log(
                        data={
                            "train_sdr": sdrs["train"],
                            "test_sdr": sdrs["test"],
                            "loss": loss.item(),
                        },
                        step=step
                    )

        # Save model.
        if step % save_step_frequency == 0:

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:

                unwrapped_model = accelerator.unwrap_model(model)

                checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
                torch.save(unwrapped_model.state_dict(), checkpoint_path)
                print("Save model to {}".format(checkpoint_path))

                checkpoint_path = Path(checkpoints_dir, "latest.pth")
                torch.save(unwrapped_model.state_dict(), Path(checkpoint_path))
                print("Save model to {}".format(checkpoint_path))

        if step == training_steps:
            break
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="UNet")
    args = parser.parse_args()

    train(args)