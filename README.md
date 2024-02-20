# A minimal Pytorch implementation of music source separation

This codebase provide a minimal Pytorch tutorial of music source separation, including examples of audio data loader, models, training, and inference pipelines.

## 0. Install dependencies.

```bash
git clone https://github.com/qiuqiangkong/mini_mss

# Install Python environment
conda create --name mss python=3.8

conda activate mss

# Install Python packages dependencies.
sh env.sh
```

## 1. Download datasets

MUSDB18HQ is a music dataset with individual vocals, bass, drums, and other stems of music. We provide a mini version which contains 1% of the full dataset for users to quickly run the code.

```bash
mkdir datasets
cd datasets

wget -O mini_musdb18hq.tar https://huggingface.co/datasets/qiuqiangkong/mini_audio_datasets/resolve/main/mini_musdb18hq.tar?download=true

tar -xvf mini_musdb18hq.tar
cd ..
```

Users can visit https://zenodo.org/records/3338373 to download the full MUSDB18HQ dataset

## 2. Train the source separation system.
```python
python train.py
```

After training on a single GPU for a few minutes, users can use the "latest.pth" checkpoint to inference their favorite songs.

## 3. Inference
```
python inference.py
```

After training for around 10 - 20 minutes (20 epochs), the vocal SDR will improve from -6 dB to around 2 dB. 

## Reference
[1] Qiuqiang Kong, Yin Cao, Haohe Liu, Keunwoo Choi, and Yuxuan Wang. "Decoupling magnitude and phase estimation with deep resunet for music source separation." arXiv preprint arXiv:2109.05418 (2021).