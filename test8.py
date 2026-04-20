import numpy as np
import librosa  # 仅用于加载音频，可替换成 wave 模块
import soundfile as sf

def cqt_kernels(fmin, n_bins, bins_per_octave, sr):
    """
    构建 CQT 复指数窗函数 (分析窗)
    返回: kernels list, 每个 kernel 对应一个 bin
    """
    Q = 1 / (2 ** (1/bins_per_octave) - 1)
    kernels = []
    for k in range(n_bins):
        fk = fmin * 2 ** (k / bins_per_octave)
        N = int(np.ceil(Q * sr / fk))  # 窗长度
        n = np.arange(N)
        window = np.hanning(N)
        kernel = window * np.exp(-2j * np.pi * fk * n / sr)
        kernel = kernel / np.linalg.norm(kernel)  # 归一化
        kernels.append(kernel)
    return kernels

def cqt_forward(x, kernels, hop_length):
    """CQT分析"""
    n_bins = len(kernels)
    n_frames = int(np.ceil(len(x) / hop_length))
    X = np.zeros((n_bins, n_frames), dtype=np.complex64)
    
    for k, kernel in enumerate(kernels):
        Nk = len(kernel)
        for m in range(n_frames):
            start = m * hop_length
            end = start + Nk
            if end > len(x):
                x_pad = np.zeros(Nk)
                x_pad[:len(x[start:])] = x[start:]
            else:
                x_pad = x[start:end]
            X[k, m] = np.sum(x_pad * np.conj(kernel))
    return X

def cqt_inverse(X, kernels, hop_length, signal_len):
    """CQT重构 (dual window 法)"""
    n_bins, n_frames = X.shape
    x_recon = np.zeros(signal_len)
    norm = np.zeros(signal_len)
    
    for k, kernel in enumerate(kernels):
        Nk = len(kernel)
        # dual window 简化：直接用 kernel 能量归一化
        dual = kernel / np.sum(np.abs(kernel) ** 2)
        for m in range(n_frames):
            start = m * hop_length
            end = start + Nk
            if end > signal_len:
                x_recon[start:] += (X[k, m] * dual[:signal_len-start]).real
                norm[start:] += np.abs(dual[:signal_len-start])**2
            else:
                x_recon[start:end] += (X[k, m] * dual).real
                norm[start:end] += np.abs(dual)**2
    # 避免除零
    norm[norm==0] = 1.0
    x_recon /= norm
    return x_recon

# -----------------------------
# 示例用法
# -----------------------------
# 1. 载入音频
y, sr = librosa.load('assets/music_10s.wav', sr=None)

# 2. CQT参数
fmin = 32.7
n_bins = 84
bins_per_octave = 12
hop_length = 128

# 3. 构建CQT kernel
kernels = cqt_kernels(fmin, n_bins, bins_per_octave, sr)

# 4. 分析
X_cqt = cqt_forward(y, kernels, hop_length)

# 5. 重构
y_recon = cqt_inverse(X_cqt, kernels, hop_length, len(y))

# 6. 保存重构音频
sf.write('y_recon.wav', y_recon, sr)

# 7. 检查误差
error = np.max(np.abs(y - y_recon))
print(f'Max reconstruction error: {error:.6f}')

import matplotlib.pyplot as plt
plt.matshow(X_cqt, origin='lower', aspect='auto', cmap='jet')
plt.savefig("_zz.pdf")
from IPython import embed; embed(using=False); os._exit(0)