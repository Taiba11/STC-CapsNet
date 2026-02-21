# Detailed Experimental Results

## Performance Metrics (Table I from paper)

| Metric | FoR (Mel) | FoR (Grayscale) | ASVspoof 2019 (Mel) | ASVspoof 2019 (Grayscale) |
|--------|-----------|-----------------|--------------------|-----------------------|
| Accuracy | **98.5%** | 95.9% | 93.4% | 90.2% |
| Precision | **98.9%** | 94.3% | 92.8% | 89.5% |
| Recall | **97.8%** | 93.5% | 91.7% | 88.7% |
| F1-Score | **98.4%** | 93.9% | 92.3% | 89.1% |
| EER | **2.8%** | 5.3% | 5.5% | 8.1% |
| Training Time | 7.5 hours | 5.8 hours | — | — |

---

## Ablation Study (Table II from paper)

| Configuration | Mel Accuracy | Grayscale Accuracy |
|--------------|-------------|-------------------|
| Full Model | **98.5%** | **95.9%** |
| Without Temporal Convolution (1D) | 94.2% (−4.3%) | 91.1% (−4.8%) |
| Without Spectral Convolution (2D) | 92.5% (−6.0%) | 89.4% (−6.5%) |
| Without Dynamic Routing | 95.1% (−3.4%) | 92.0% (−3.9%) |
| With Reduced Capsules | 96.0% (−2.5%) | 92.8% (−3.1%) |

**Key Findings:**
- Spectral convolution removal causes the largest accuracy drop → frequency features are critical
- Temporal convolution is essential for detecting timing anomalies
- Dynamic routing significantly improves capsule communication
- Full capsule capacity needed for detailed time-frequency interaction modeling

---

## Comparison with State-of-the-Art (Table III from paper)

| Study | Features | Model | Dataset | F1-Score | EER (%) |
|-------|----------|-------|---------|----------|---------|
| Ustubioglu et al. (TSP 2024) | Cochleagram | ArCapsNet | TIMIT | 98.29% | — |
| Mao et al. (FCS 2021) | MFCC | Capsule Net | ASVspoof 2019 | — | 9.21 |
| Mao et al. (FCS 2021) | CQCC | Capsule Net | ASVspoof 2019 | — | 5.09 |
| **STC-CapsNet (Ours)** | **Mel-spectrogram** | **STC-CapsNet** | **FoR** | **98.4%** | **2.8** |
| STC-CapsNet (Ours) | Grayscale | STC-CapsNet | FoR | 93.9% | 5.3 |

---

## Training Configuration (Section III-B)

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| LR Scheduler | ReduceOnPlateau (factor=0.1, patience=5) |
| Batch Size | 32 |
| Epochs | 100 (with early stopping, patience=10) |
| Loss | Margin Loss (m+=0.9, m−=0.1, λ=0.5) |
| Data Split | 70% train / 15% val / 15% test |
| Augmentation | Time-shifting, Pitch-shifting |
| Routing Iterations | 3 |
