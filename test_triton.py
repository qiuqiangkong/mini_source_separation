import torch
import triton
import triton.language as tl


@triton.jit
def simple_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr
):
    pid = tl.program_id(0)  # block id
    row = pid  # 每个 block 处理一行
    if row >= M:
        return

    acc = 0.0
    for k in range(K):
        a = tl.load(A_ptr + row * K + k)
        b = tl.load(B_ptr + k * N + row % N)  # 简单示例
        acc += a * b
    tl.store(C_ptr + row * N + row % N, acc)


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)   # block id in M dimension
    pid_n = tl.program_id(1)   # block id in N dimension

    # block 的起始位置
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 创建结果 tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K方向多次累加
    for k in range(0, K, BLOCK_K):
        # A tile: (BLOCK_M, BLOCK_K)
        a = tl.load(A + (offs_m[:, None] * stride_am
                         + (k + offs_k)[None, :] * stride_ak),
                    mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
                    other=0.0)

        # B tile: (BLOCK_K, BLOCK_N)
        b = tl.load(B + ((k + offs_k)[:, None] * stride_bk
                         + offs_n[None, :] * stride_bn),
                    mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N),
                    other=0.0)

        acc += tl.dot(a, b)

    # 写回 C
    tl.store(
        C + (offs_m[:, None] * stride_cm
             + offs_n[None, :] * stride_cn),
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


def add():

    # 测试数据
    M, N, K = 4, 4, 4
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    C = torch.zeros(M, N, device='cuda', dtype=torch.float32)

    # Launch kernel
    grid = (M,)  # 每个 block 对应一行
    simple_matmul_kernel[grid](A, B, C, M, N, K)

    # 打印结果
    print("A:\n", A)
    print("B:\n", B)
    print("C:\n", C)
    print("Reference C = A @ B:\n", A @ B)


def add2():

    a = torch.randn(128, 256, device='cuda', dtype=torch.float32)
    b = torch.randn(256, 64, device='cuda', dtype=torch.float32)
    # c = triton_matmul(a, b)
    

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    BLOCK_M=32
    BLOCK_N=32
    BLOCK_K=32
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    print(torch.allclose(c, a @ b, atol=1e-2))
    


if __name__ == '__main__':

    add2()