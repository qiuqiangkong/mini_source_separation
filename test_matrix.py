import torch
from torch import Tensor
import numpy as np
import time


def add():
    device = "cuda"
    N = 1000000
    D1 = 1000
    D2 = 1000
    rs = np.random.RandomState(1234)
    A = Tensor(rs.uniform(size=(N, D1))).to(device)
    B = Tensor(rs.uniform(size=(D1, D2))).to(device)

    # for _ in range(20):
    while True:
        t1 = time.time()
        A @ B
        print(time.time() - t1)


def add2():
    device = "cuda"
    N = 1000000
    D1 = 1000
    D2 = 1000
    rs = np.random.RandomState(1234)
    A = Tensor(rs.uniform(size=(N, D1))).to(device)
    B = Tensor(rs.uniform(size=(D1, D2))).to(device)

    # for _ in range(20):
    while True:
        t1 = time.time()
        
        i = 0
        chunk_size = 100
        while i < N:
            A[i : i + chunk_size] @ B
            i += chunk_size


        print(time.time() - t1)


@torch.compile
def func3(A, B):
    i = 0
    N = A.shape[0]
    chunk_size = 100
    while i < N:
        A[i : i + chunk_size] @ B
        i += chunk_size


def add3():
    device = "cuda"
    N = 1000000
    D1 = 1000
    D2 = 1000
    rs = np.random.RandomState(1234)
    A = Tensor(rs.uniform(size=(N, D1))).to(device)
    B = Tensor(rs.uniform(size=(D1, D2))).to(device)

    # for _ in range(20):
    while True:
        t1 = time.time()
        
        func3(A, B)

        print(time.time() - t1)


if __name__ == '__main__':
    add3()