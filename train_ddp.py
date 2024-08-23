import argparse
import os
import random
from pathlib import Path
from typing import Optional, Union, Literal
import yaml
import torch
import librosa
import museval
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb

wandb.require("core")

from roformer_dataset import load
from roformer_dataset import MUSDB18HQ
from roformer_dataset import RandomCrop

def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

def setup(rank, world_size):
    dist.init_process_group(backend="nccl", 
                            rank=rank, 
                            world_size=world_size,
                            init_method="tcp://localhost:54321")
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def separate(
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

    clips = librosa.util.frame(
        audio, 
        frame_length=clip_samples, 
        hop_length=clip_samples
    )
    # shape: (channels_num, clip_samples, clips_num)
    
    clips = clips.transpose(2, 0, 1)
    # shape: (clips_num, channels_num, clip_samples)

    clips_num = clips.shape[0]

    pointer = 0
    outputs = []

    while pointer < clips_num:

        batch_clips = torch.Tensor(clips[pointer : pointer + batch_size].copy()).to(device)

        with torch.no_grad():
            model.eval()
            batch_output = model(batch_clips)
            batch_output = batch_output.cpu().numpy()

        outputs.append(batch_output)
        pointer += batch_size

    outputs = np.concatenate(outputs, axis=0)
    # shape: (clips_num, channels_num, clip_samples)

    channels_num = outputs.shape[1]
    outputs = outputs.transpose(1, 0, 2).reshape(channels_num, -1)
    # shape: (channels_num, clips_num * clip_samples)

    outputs = outputs[:, 0 : audio_samples]
    # shape: (channels_num, audio_samples)

    return outputs

def get_model(model_name, config=None):
    if model_name == "UNet":
        from models.unet import UNet
        return UNet()
        # pass
    elif model_name == "BSRoformer":
        from models.bs_roformer import BSRoformer
        return BSRoformer(
            depth=12,
            dim=384,
            n_heads=12,
            # attn_dropout=0.1,
            # ff_dropout=0.1,
        )
    elif model_name == "BSRoformer2":
        from models.bs_roformer2 import BSRoformer2
        return BSRoformer2(
            depth=12,
            dim=384,
            n_heads=12,
        )
        # pass
    elif model_name == "BSRoformer_uvr":
        from models.bs_roformer_uvr import BSRoformer
        assert config != None
        return BSRoformer(
            **config
        )
    else:
        raise NotImplementedError
    
class InfiniteSampler:
    def __init__(self, dataset):

        self.indexes = list(range(len(dataset)))
        random.shuffle(self.indexes)
        
    def __iter__(self):

        pointer = 0

        while True:

            if pointer == len(self.indexes):
                random.shuffle(self.indexes)
                pointer = 0
                
            index = self.indexes[pointer]
            pointer += 1

            yield index


def warmup_lambda(step, warm_up_steps=1000):
    if step <= warm_up_steps:
        return step / warm_up_steps
    else:
        return 1.


def l1_loss(output, target):
    return torch.mean(torch.abs(output - target))



def validate(
    root: str, 
    split: Literal["train", "test"], 
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

        sep_wav = separate(
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

    sdr = np.nanmedian(all_sdrs)

    return sdr


def train(rank, world_size, args, model_config):

    setup(rank, world_size)

    # Arguments
    model_name = args.model_name
    clip_duration = args.clip_duration
    batch_size = args.per_device_batch_size
    lr = float(args.lr)

    # Default parameters
    sr = args.sr
    mono = args.mono
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    use_scheduler = args.use_scheduler
    test_step_frequency = args.test_step_frequency
    save_step_frequency = args.save_step_frequency
    evaluate_num = args.evaluate_num
    training_steps = args.training_steps
    num_epochs = args.num_epochs
    
    device = torch.device(f"cuda:{rank}")

    filename = Path(__file__).stem
    source_types = MUSDB18HQ.source_types

    checkpoints_dir = Path("./checkpoints", filename, model_name)
    root = "/sdb/data1/MUSDB/MUSDB18HQ"

    # TensorBoard
    if rank == 0:
        writer = SummaryWriter(log_dir=f"./tensorboard_logs/{model_name}")

    # Training dataset
    train_dataset = MUSDB18HQ(
        root=root,
        split="train",
        sr=sr,
        crop=RandomCrop(clip_duration=clip_duration, end_pad=0.)
    )

    # Samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    # Dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    # Model
    model = get_model(model_name, config=model_config)
    model.load_state_dict(torch.load(args.checkpoints_dir, map_location='cpu'))
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    if use_scheduler:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warmup_lambda)

    # Create checkpoints directory
    if rank == 0:
        Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        for step, data in enumerate(tqdm(train_dataloader)):

            mixture = data["mixture"].to(device)
            target = data["vocals"].to(device)

            # Forward
            model.train()
            output = model(mixture) 
            
            # Calculate loss
            loss = l1_loss(output, target)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Learning rate scheduler (optional)
            if use_scheduler:
                scheduler.step()
            
            # Evaluate
            if step % test_step_frequency == 0 and rank == 0:
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
                        model=model,
                        evaluate_num=evaluate_num,
                    )
                    sdrs[split] = sdr

                print(f"--- Epoch: {epoch}, Step: {step} ---")
                print(f"Loss: {loss:.3f}")
                print(f"Train SDR: {sdrs['train']:.3f}")
                print(f"Test SDR: {sdrs['test']:.3f}")

                # Log to TensorBoard
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + step)
                writer.add_scalar('SDR/train', sdrs['train'], epoch * len(train_dataloader) + step)
                writer.add_scalar('SDR/test', sdrs['test'], epoch * len(train_dataloader) + step)

            # Save model.
            if step % save_step_frequency == 0 and rank == 0:
                checkpoint_path = Path(checkpoints_dir, f"step={step}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Save model to {checkpoint_path}")

                checkpoint_path = Path(checkpoints_dir, "latest.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Save model to {checkpoint_path}")

            if step == training_steps:
                break

    if rank == 0:
        writer.close()

    cleanup()





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="BSRoformer_uvr")
    parser.add_argument('--model_config_path', type=str, default="/home/zhiyuew/my_bs_roformer/model_bs_roformer_ep_317_sdr_12.9755.yaml")
    parser.add_argument('--checkpoints_dir', type=str, default="/home/zhiyuew/my_bs_roformer/model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    parser.add_argument('--clip_duration', type=float, default=2.0)
    parser.add_argument('--per_device_batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--mono", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--use_scheduler", type=bool, default=True)
    parser.add_argument("--test_step_frequency", type=int, default=5000)
    parser.add_argument("--save_step_frequency", type=int, default=10000)
    parser.add_argument("--evaluate_num", type=int, default=10)
    parser.add_argument("--training_steps", type=int, default=1000000)
    parser.add_argument("--num_epochs", type=int, default=100000)
    args = parser.parse_args()
    
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)
    with open(args.model_config_path, 'r') as f:
        config = yaml.safe_load(f)

    world_size = torch.cuda.device_count()

    mp.spawn(
        train,
        args=(world_size, args, config['model']),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
