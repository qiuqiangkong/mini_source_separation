import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SequentialSampler
import numpy as np
import soundfile
from pathlib import Path
import torch.optim as optim
from tqdm import tqdm
import argparse
import random
from accelerate import Accelerator
import wandb
wandb.require("core")

from data.musdb18hq import MUSDB18HQ
from models.unet import UNet


def train(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    sr = 44100
    batch_size = 16
    num_workers = 16
    pin_memory = True
    learning_rate = 1e-3
    use_scheduler = True
    test_step_frequency = 1000
    save_step_frequency = 1000
    training_steps = 100000
    debug = False
    wandb_log = True
    device = "cuda"

    filename = Path(__file__).stem

    if wandb_log:
        wandb.init(project="mini_source_separation")

    checkpoints_dir = Path("./checkpoints", model_name)
    
    root = "/datasets/musdb18hq"

    # Dataset
    train_dataset = MUSDB18HQ(
        root=root,
        split="train",
        sr=44100,
        mono=False,
        segment_duration=2.,
    )

    test_dataset = MUSDB18HQ(
        root=root,
        split="test",
        sr=44100,
        mono=False,
        segment_duration=2.,
    )

    # Sampler
    train_sampler = InfiniteSampler(train_dataset)
    test_sampler = SequentialSampler(test_dataset)

    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    eval_train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=test_sampler,
        num_workers=1, 
        pin_memory=pin_memory
    )

    eval_test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        sampler=test_sampler,
        num_workers=1, 
        pin_memory=pin_memory
    )

    # Model
    model = get_model(model_name)
    model.to(device)

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

        mixture = data["mixture"].to(device)
        target = data["vocals"].to(device)

        # Play the audio
        if debug:
            play_audio(mixture, target)

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
                
                train_sdr = validate(val_model, eval_train_dataloader)
                test_sdr = validate(val_model, eval_test_dataloader)

                print("--- step: {} ---".format(step))
                print("Loss: {:.3f}".format(loss))
                print("Train SDR: {:.3f}".format(train_sdr))
                print("Test SDR: {:.3f}".format(test_sdr))

                if wandb_log:
                    wandb.log(
                        data={
                            "train_sdr": train_sdr,
                            "test_sdr": test_sdr,
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
        


def get_model(model_name):
    if model_name == "UNet":
        return UNet()
    else:
        raise NotImplementedError


class InfiniteSampler:
    def __init__(self, dataset):

        self.indexes = list(range(len(dataset)))
        random.shuffle(self.indexes)
        
    def __iter__(self):

        i = 0

        while True:

            if i == len(self.indexes):
                random.shuffle(self.indexes)
                i = 0
                
            index = self.indexes[i]
            i += 1

            yield index


def warmup_lambda(step, warm_up_steps=1000):
    if step <= warm_up_steps:
        return step / warm_up_steps
    else:
        return 1.


def l1_loss(output, target):
    return torch.mean(torch.abs(output - target))


def play_audio(mixture, target):
    soundfile.write(file="tmp_mixture.wav", data=mixture[0].cpu().numpy().T, samplerate=44100)
    soundfile.write(file="tmp_target.wav", data=target[0].cpu().numpy().T, samplerate=44100)
    from IPython import embed; embed(using=False); os._exit(0)


def validate(model, dataloader):

    device = next(model.parameters()).device

    sdrs = []

    for step, data in tqdm(enumerate(dataloader)):

        mixture = data["mixture"].to(device)
        target = data["vocals"].to(device)

        with torch.no_grad():
            model.eval()
            output = model(mixture=mixture)

        target = target.cpu().numpy()
        output = output.cpu().numpy()

        for tar, out in zip(target, output):
            sdr = calculate_sdr(ref=tar, est=out)
            sdrs.append(sdr)

    return np.nanmedian(sdrs)


def calculate_sdr(ref, est):
    eps = 1e-12
    s_true = ref
    s_artif = est - ref
    numerator = np.sum(s_true ** 2) + eps
    denominator = np.sum(s_artif ** 2) + eps
    sdr = 10. * (np.log10(numerator) - np.log10(denominator))

    return sdr


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="UNet")
    args = parser.parse_args()

    train(args)