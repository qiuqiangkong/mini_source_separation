import torch
import triton
import triton.language as tl

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
    """
    C = A @ B
    A: [M, K]
    B: [K, N]
    C: [M, N]
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute start offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load tiles from A and B
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # Mask for out-of-bounds
        mask_a = offs_m[:, None] < M
        mask_k_a = offs_k[None, :] < K
        mask_a = mask_a & mask_k_a
        mask_b = offs_k[:, None] < K
        mask_n_b = offs_n[None, :] < N
        mask_b = mask_b & mask_n_b

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # dot product
        acc += tl.dot(a, b)

    # Store result
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = offs_m[:, None] < M
    mask_n_c = offs_n[None, :] < N
    mask_c = mask_c & mask_n_c
    tl.store(c_ptrs, acc, mask=mask_c)

    # tl.device_print("a", c_ptrs[0])


def triton_matmul(A, B, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    from IPython import embed; embed(using=False); os._exit(0)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    return C


def add():
    # M, N, K = 128, 256, 512
    M, N, K = 153, 221, 420
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)

    C = triton_matmul(A, B)

    # 验证结果
    C_ref = A @ B
    print((C - C_ref).abs().mean())
    # print(torch.allclose(C, C_ref, rtol=1e-2, atol=1e-2))  # True


def add2():
    pass


if __name__ == '__main__':
    add()