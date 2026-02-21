<div align="center">

# STC-CapsNet: Detecting Audio Deepfakes with Spatio-Temporal Convolutions and Capsule Networks

[![Paper](https://img.shields.io/badge/Paper-IEEE%20CISM%202025-blue.svg)](https://doi.org/10.1109/CISM64958.2025.11060861)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![Conference](https://img.shields.io/badge/IEEE%20CISM-2025-purple.svg)](#)

**Official implementation of the paper accepted at IEEE Symposium on Computational Intelligence in Image, Signal Processing and Synthetic Media (CISM) 2025**

[Taiba Majid Wani](mailto:majid@diag.uniroma1.it)<sup>1</sup>&nbsp;&nbsp;&nbsp;
[Syed Asif Ahmad Qadri](mailto:syedasif@m110.nthu.edu.tw)<sup>2</sup>&nbsp;&nbsp;&nbsp;
[Irene Amerini](mailto:amerini@diag.uniroma1.it)<sup>1</sup>

<sup>1</sup>Sapienza University of Rome, Italy &nbsp;&nbsp; <sup>2</sup>National Tsing Hua University, Taiwan

<br>

<img src="assets/architecture.png" alt="STC-CapsNet Architecture" width="850"/>

</div>

---

## 📋 Abstract

Capsule networks are a powerful architecture designed to capture hierarchical relationships in data, making them effective for complex classification tasks. This study introduces the novel **Spatio-Temporal Convolutional Capsule Network (STC-CapsNet)**, which utilizes mel-spectrograms and grayscale spectrograms for feature extraction. After preprocessing steps like noise reduction and segmentation, **temporal (1D) and spectral (2D) convolutions** are applied, followed by **capsule layers with dynamic routing** to enhance feature representation. The model is evaluated on the FoR dataset, achieving an **F1-Score of 98.4%** and a low **EER of 2.8%** using mel-spectrograms.

### 🏆 Key Results

| Dataset | Feature | Accuracy | F1-Score | EER (%) |
|---------|---------|----------|----------|---------|
| **FoR** | Mel-spectrogram | **98.5%** | **98.4%** | **2.8** |
| **FoR** | Grayscale spectrogram | 95.9% | 93.9% | 5.3 |
| **ASVspoof 2019** (cross-dataset) | Mel-spectrogram | 93.4% | 92.3% | 5.5 |
| **ASVspoof 2019** (cross-dataset) | Grayscale spectrogram | 90.2% | 89.1% | 8.1 |

---

## 🔥 Highlights

- **Dual Spectrogram Support** — Supports both mel-spectrograms and grayscale spectrograms for flexible feature extraction
- **Spatio-Temporal Convolutions** — 1D temporal convolutions capture time-domain dependencies; 2D spectral convolutions capture frequency-domain patterns
- **Capsule Networks with Dynamic Routing** — Preserves detailed time-frequency relationships lost by traditional CNNs
- **Cross-Dataset Generalization** — Validated on FoR (primary) and ASVspoof 2019 (cross-dataset)

---

## 🏗️ Architecture

```
Audio Input
    │
    ▼
┌─────────────────────┐
│   Preprocessing      │  Noise Reduction → Segmentation → Silence Removal
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌──────────────┐
│   Mel   │ │  Grayscale   │   Two feature extraction paths
│ Spectro │ │  Spectrogram │
└────┬────┘ └──────┬───────┘
     │             │
     └─────┬───────┘
           ▼
┌─────────────────────┐
│  1D Temporal Conv    │  Captures short/long-term time dependencies
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  2D Spectral Conv    │  Captures localized frequency patterns
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Time-Frequency      │  Combined spatio-temporal features
│  Features            │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Primary Capsules    │  Encode time-frequency relationships as vectors
└──────────┬──────────┘
           │ Dynamic Routing
           ▼
┌─────────────────────┐
│  Higher Capsules     │  Aggregate features via routing-by-agreement
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Classification      │  Margin Loss → Real / Fake
└─────────────────────┘
```

---

## 📁 Project Structure

```
STC-CapsNet/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── setup.py
├── .gitignore
│
├── configs/
│   ├── default.yaml              # Default training configuration
│   ├── mel_spectrogram.yaml      # Mel-spectrogram specific config
│   └── grayscale_spectrogram.yaml # Grayscale spectrogram config
│
├── datasets/
│   ├── __init__.py
│   ├── for_dataset.py            # FoR dataset loader
│   ├── asvspoof2019.py           # ASVspoof2019 cross-dataset loader
│   └── preprocessing.py          # Preprocessing & spectrogram generation
│
├── models/
│   ├── __init__.py
│   ├── stc_capsnet.py            # Full STC-CapsNet architecture
│   ├── temporal_conv.py          # 1D temporal convolution module
│   ├── spectral_conv.py          # 2D spectral convolution module
│   ├── capsule_layers.py         # Primary & higher capsule layers
│   └── losses.py                 # Margin loss implementation
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py                # EER, accuracy, F1, precision, recall
│   ├── logger.py                 # Training logger with TensorBoard
│   └── visualization.py          # Spectrogram & results visualization
│
├── scripts/
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── inference.py              # Single-file inference
│   └── generate_spectrograms.py  # Batch spectrogram generation
│
├── assets/
│   └── architecture.png          # Architecture diagram
│
└── docs/
    └── RESULTS.md                # Detailed experimental results
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/CapsuleNetworks/STC-CapsNet.git
cd STC-CapsNet

# Create virtual environment
conda create -n stccapsnet python=3.9 -y
conda activate stccapsnet

# Install dependencies
pip install -r requirements.txt

# (Optional) Install as package
pip install -e .
```

---

## 📊 Dataset Preparation

### FoR Dataset

1. Download from the [FoR dataset page](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
2. Organize as:

```
data/
└── FoR/
    ├── for-original/
    ├── for-norm/
    ├── for-2seconds/
    └── for-rerecorded/
```

### ASVspoof 2019 (Cross-Dataset Evaluation)

1. Download from the [ASVspoof website](https://www.asvspoof.org/index2019.html)
2. Organize as:

```
data/
└── ASVspoof2019/
    └── LA/
        ├── ASVspoof2019_LA_train/
        ├── ASVspoof2019_LA_dev/
        ├── ASVspoof2019_LA_eval/
        └── ASVspoof2019_LA_cm_protocols/
```

### Generate Spectrograms

```bash
# Mel-spectrograms
python scripts/generate_spectrograms.py \
    --data_dir data/FoR \
    --output_dir data/spectrograms/for_mel \
    --feature_type mel

# Grayscale spectrograms
python scripts/generate_spectrograms.py \
    --data_dir data/FoR \
    --output_dir data/spectrograms/for_gray \
    --feature_type grayscale
```

---

## 🚀 Training

### Train with Mel-Spectrograms (Recommended)

```bash
python scripts/train.py \
    --config configs/mel_spectrogram.yaml \
    --data_dir data/FoR \
    --output_dir experiments/for_mel \
    --feature_type mel \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001
```

### Train with Grayscale Spectrograms

```bash
python scripts/train.py \
    --config configs/grayscale_spectrogram.yaml \
    --data_dir data/FoR \
    --output_dir experiments/for_gray \
    --feature_type grayscale \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001
```

---

## 📈 Evaluation

```bash
# Evaluate on FoR test set
python scripts/evaluate.py \
    --checkpoint experiments/for_mel/best_model.pth \
    --data_dir data/FoR \
    --feature_type mel

# Cross-dataset evaluation on ASVspoof 2019
python scripts/evaluate.py \
    --checkpoint experiments/for_mel/best_model.pth \
    --data_dir data/ASVspoof2019/LA \
    --dataset asvspoof2019 \
    --feature_type mel
```

### Single File Inference

```bash
python scripts/inference.py \
    --checkpoint experiments/for_mel/best_model.pth \
    --audio_path path/to/audio.wav \
    --feature_type mel
```

---

## 📊 Results

### Performance Comparison: Mel vs. Grayscale Spectrograms

| Metric | FoR (Mel) | FoR (Gray) | ASVspoof2019 (Mel) | ASVspoof2019 (Gray) |
|--------|-----------|------------|-------------------|-------------------|
| Accuracy | **98.5%** | 95.9% | 93.4% | 90.2% |
| Precision | **98.9%** | 94.3% | 92.8% | 89.5% |
| Recall | **97.8%** | 93.5% | 91.7% | 88.7% |
| F1-Score | **98.4%** | 93.9% | 92.3% | 89.1% |
| EER | **2.8%** | 5.3% | 5.5% | 8.1% |

### Ablation Study (FoR Dataset)

| Configuration | Mel Accuracy | Grayscale Accuracy |
|--------------|-------------|-------------------|
| Full Model | **98.5%** | **95.9%** |
| w/o Temporal Convolution | 94.2% | 91.1% |
| w/o Spectral Convolution | 92.5% | 89.4% |
| w/o Dynamic Routing | 95.1% | 92.0% |
| Reduced Capsules | 96.0% | 92.8% |

### Comparison with State-of-the-Art

| Method | Features | Model | Dataset | F1-Score | EER (%) |
|--------|----------|-------|---------|----------|---------|
| Ustubioglu et al. | Cochleagram | ArCapsNet | TIMIT | 98.29% | — |
| Mao et al. | MFCC | Capsule Net | ASVspoof 2019 | — | 9.21 |
| Mao et al. | CQCC | Capsule Net | ASVspoof 2019 | — | 5.09 |
| **STC-CapsNet (Ours)** | **Mel-spectrogram** | **STC-CapsNet** | **FoR** | **98.4%** | **2.8** |
| **STC-CapsNet (Ours)** | Grayscale | STC-CapsNet | FoR | 93.9% | 5.3 |

---

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{wani2025stccapsnet,
    title     = {STC-CapsNet: Detecting Audio Deepfakes with Spatio-Temporal Convolutions and Capsule Networks},
    author    = {Wani, Taiba Majid and Qadri, Syed Asif Ahmad and Amerini, Irene},
    booktitle = {2025 IEEE Symposium on Computational Intelligence in Image, Signal Processing and Synthetic Media (CISM)},
    year      = {2025},
    doi       = {10.1109/CISM64958.2025.11060861},
    publisher = {IEEE}
}
```

---

## 🙏 Acknowledgments

This study has been partially supported by:
- **SERICS** (PE00000014) under the MUR National Recovery and Resilience Plan funded by the European Union – NextGenerationEU
- **Sapienza University of Rome** project 2022–2024 "EV2" (003 009 22)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you find this repository helpful, please consider giving it a star! ⭐**

</div>
