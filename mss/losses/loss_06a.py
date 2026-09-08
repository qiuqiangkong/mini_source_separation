from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Loss06a(nn.Module):
    def __init__(self):
        super().__init__()

    '''
    def forward(self, output: Tensor, target: Tensor) -> torch.float:

        window_size = 2048
        hop_length = 480

        B = output.shape[0]
        out = rearrange(output, 'b c l -> (b c) l')
        tgt = rearrange(target, 'b c l -> (b c) l')

        out = self.enframe(out, window_size, hop_length)
        tgt = self.enframe(tgt, window_size, hop_length)

        eps = 1e-10
        a1 = torch.clamp((tgt ** 2).mean(dim=-1), eps)
        a1 = torch.log10(a1)

        b1 = torch.clamp(((out - tgt) ** 2).mean(dim=-1), eps)
        b1 = torch.log10(b1)

        sdr = a1 - b1
        loss = - sdr.mean()

        # loss *= 10
        print(loss)
        return loss
    '''
    def forward(self, output: Tensor, target: Tensor) -> torch.float:

        window_size = 2048
        hop_length = 480

        B = output.shape[0]
        out = rearrange(output, 'b c l -> (b c) l')
        tgt = rearrange(target, 'b c l -> (b c) l')

        out = self.enframe(out, window_size, hop_length)
        tgt = self.enframe(tgt, window_size, hop_length)

        eps = 1e-10
        # a1 = torch.clamp((tgt ** 2).mean(dim=-1), eps)
        # a1 = torch.log10(a1)
        # from IPython import embed; embed(using=False); os._exit(0)

        # b1 = torch.clamp(((out - tgt) ** 2).mean(dim=-1), eps)
        b1 = ((out - tgt) ** 2).mean(dim=-1)
        b1 = torch.log10(b1 + 1e-6)
        # b1 = torch.log(b1 + 1e-8)

        # sdr = a1 - b1
        # loss = - sdr.mean()
        loss = b1.mean()

        # loss *= 10
        # print(loss)
        return loss
    '''
    def forward(self, output: Tensor, target: Tensor) -> torch.float:

        window_size = 2048
        hop_length = 480

        B = output.shape[0]
        out = rearrange(output, 'b c l -> (b c) l')
        tgt = rearrange(target, 'b c l -> (b c) l')

        out = self.enframe(out, window_size, hop_length)
        tgt = self.enframe(tgt, window_size, hop_length)

        eps = 1e-10
        # a1 = torch.clamp((tgt ** 2).mean(dim=-1), eps)
        # a1 = torch.log10(a1)

        b1 = torch.clamp(((out - tgt) ** 2).mean(dim=-1), eps)
        # b1 = torch.log10(b1)

        # sdr = a1 - b1
        # loss = - sdr.mean()
        loss = b1.mean()

        # loss *= 10
        print(loss)
        return loss
    '''
    def enframe(self, x, window_size, hop_length):

        N = window_size
        x = F.pad(x, (N // 2, N // 2), mode="reflect")  # (b, l)
        x = x.unfold(dimension=-1, size=N, step=hop_length).contiguous()  # (b, t, n)
        return x