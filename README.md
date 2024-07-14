# A minimal Pytorch implementation of music source separation

This codebase provide a minimal Pytorch tutorial of music source separation. We use the [MUSDB18HQ](https://zenodo.org/records/3338373) dataset containing 100 training and 50 testing songs for training and validation, respectively.

## 0. Download datasets

Users need to download the dataset from https://zenodo.org/records/3338373. After download and unzip, the dataset looks like:

<pre>
dataset_root (30 GB)
	├── train (100 files)
	│   ├── A Classic Education - NightOwl
	│   │   ├── bass.wav
	│   │   ├── drums.wav
	│   │   ├── mixture.wav
	│   │   ├── other.wav
	│   │   └── vocals.wav
	│		... 
	│		└── ...
	└── test (50 files)
	    ├── Al James - Schoolboy Facination
	    │   ├── bass.wav
	    │   ├── drums.wav
	    │   ├── mixture.wav
	    │   ├── other.wav
	    │   └── vocals.wav
		 	... 
		 	└── ...
</pre>

## 1. Install dependencies.

```bash
git clone https://github.com/qiuqiangkong/mini_mss

# Install Python environment
conda create --name mss python=3.8

conda activate mss

# Install Python packages dependencies.
sh env.sh
```

## 2. Train the source separation system.
```python
CUDA_VISIBLE_DEVICES=0 python train.py
```

After training on a single GPU for a few minutes, users can use the "latest.pth" checkpoint to inference their favorite songs.

## 3. Inference
```python
python inference.py
```

After training for around 10 - 20 minutes (20 epochs), the vocal SDR will improve from -6 dB to around 2 dB. 

## Reference
<pre>
@inproceedings{kong2021decoupling,
  title={Decoupling magnitude and phase estimation with deep resunet for music source separation},
  author={Kong, Qiuqiang and Cao, Yin and Liu, Haohe and Choi, Keunwoo and Wang, Yuxuan},
  booktitle={International Society for Music Information Retrieval (ISMIR)},
  year={2021}
}
</pre>