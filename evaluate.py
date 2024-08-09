import argparse
from pathlib import Path

import torch

from data.musdb18hq import MUSDB18HQ
from train import get_model, validate


def evaluate(args):

    model_name = args.model_name
    checkpoint_path = args.checkpoint_path
    clip_duration = args.clip_duration
    batch_size = args.batch_size
    evaluate_num = None

    root = "/datasets/musdb18hq"
    split = "test"
    sr = 44100
    device = "cuda"
    source_types = MUSDB18HQ.source_types

    model = get_model(model_name)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

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
        verbose=True
    )

    print("--- Median SDR ---")
    print("{:.2f} dB".format(sdr))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="UNet")
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--clip_duration', type=float, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    evaluate(args)