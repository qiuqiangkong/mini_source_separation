# A minimal Pytorch implementation of music source separation

This codebase provide a tutorial of music source separation. This codebase provide examples of audio data loader, models, training, and inference pipelines.

## 0. Install dependencies.

```bash
pip install -r requirements.txt
```

## 1. Train the source separation system.
```python
python train.py
```

After training on a single GPU for a few minutes, users can use the "latest.pth" checkpoint to inference their favorite songs.

## 2. Inference
```
python inference.py
```

After training for around 10 - 20 minutes (20 epochs), the vocal SDR will be around 2 dB. 

## Reference
[1] Qiuqiang Kong, Yin Cao, Haohe Liu, Keunwoo Choi, and Yuxuan Wang. "Decoupling magnitude and phase estimation with deep resunet for music source separation." arXiv preprint arXiv:2109.05418 (2021).