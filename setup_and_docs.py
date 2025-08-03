# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stc-capsnet",
    version="1.0.0",
    author="Taiba Majid Wani, Syed Asif Ahmad Qadri, Irene Amerini",
    author_email="majid@diag.uniroma1.it",
    description="STC-CapsNet: Detecting Audio Deepfakes with Spatio-Temporal Convolutions and Capsule Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CapsuleNetworks/STC-CapsNet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
            "optuna>=2.10.0",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stc-capsnet-train=main:main",
            "stc-capsnet-evaluate=evaluate:main",
            "stc-capsnet-inference=inference:main",
        ],
    },
)


# requirements.txt
torch>=1.9.0
torchaudio>=0.9.0
librosa>=0.8.1
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.62.0
pathlib2>=2.3.6
scipy>=1.7.0
soundfile>=0.10.3
resampy>=0.2.2


# environment.yml
name: stc-capsnet
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch>=1.9.0
  - torchaudio>=0.9.0
  - numpy>=1.21.0
  - scipy>=1.7.0
  - scikit-learn>=1.0.0
  - matplotlib>=3.3.0
  - seaborn>=0.11.0
  - tqdm>=4.62.0
  - pip
  - pip:
    - librosa>=0.8.1
    - soundfile>=0.10.3
    - resampy>=0.2.2
    - optuna>=2.10.0


# docker/Dockerfile
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/logs /app/results

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for potential web interface
EXPOSE 8080

# Default command
CMD ["python", "main.py", "--help"]


# docker/docker-compose.yml
version: '3.8'

services:
  stc-capsnet-training:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    volumes:
      - ../data:/app/data
      - ../models:/app/models
      - ../logs:/app/logs
      - ../results:/app/results
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: >
      python main.py 
      --data_path /app/data/for_processed 
      --spectrogram_type mel 
      --batch_size 32 
      --epochs 100
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  stc-capsnet-inference:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    volumes:
      - ../data:/app/data
      - ../models:/app/models
      - ../results:/app/results
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: >
      python inference.py 
      --model_path /app/models/stc_capsnet.pth 
      --audio_dir /app/data/test_audio 
      --output /app/results/predictions.json
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]


# scripts/download_datasets.sh
#!/bin/bash

# Script to download and prepare datasets

echo "STC-CapsNet Dataset Preparation Script"
echo "======================================"

# Create data directory
mkdir -p data
cd data

# Download FoR dataset (if available publicly)
echo "Note: FoR dataset needs to be obtained from the original authors"
echo "Please visit: https://www.researchgate.net/publication/335906167_FoR_A_dataset_for_synthetic_speech_detection"
echo ""

# Download ASVspoof 2019 dataset
echo "Downloading ASVspoof 2019 LA dataset..."
if [ ! -d "ASVspoof2019_LA_eval" ]; then
    echo "Please download ASVspoof 2019 LA dataset from:"
    echo "https://datashare.ed.ac.uk/handle/10283/3336"
    echo "Extract to data/ASVspoof2019_LA_eval/"
fi

# Create sample dataset for testing
echo "Creating sample dataset for testing..."
cd ..
python data_preparation.py --create_sample --output_path ./data/sample --num_samples 100

echo "Dataset preparation instructions completed!"
echo "Please manually download the required datasets as indicated above."


# scripts/train_all_variants.sh
#!/bin/bash

# Script to train all model variants

echo "Training STC-CapsNet variants..."

# Create directories
mkdir -p models logs results

# Train with mel-spectrograms
echo "Training with mel-spectrograms..."
python main.py \
    --data_path ./data/for_processed \
    --spectrogram_type mel \
    --batch_size 32 \
    --epochs 100 \
    --save_model models/stc_capsnet_mel.pth \
    --log_dir logs/mel

# Train with grayscale spectrograms
echo "Training with grayscale spectrograms..."
python main.py \
    --data_path ./data/for_processed \
    --spectrogram_type grayscale \
    --batch_size 32 \
    --epochs 100 \
    --save_model models/stc_capsnet_grayscale.pth \
    --log_dir logs/grayscale

# Run ablation study
echo "Running ablation study..."
python train_ablation.py

# Cross-dataset evaluation
echo "Cross-dataset evaluation..."
if [ -d "./data/ASVspoof2019_LA_eval" ]; then
    python cross_dataset_evaluation.py \
        --source_model models/stc_capsnet_mel.pth \
        --target_dataset ./data/ASVspoof2019_LA_eval \
        --spectrogram_type mel
        
    python cross_dataset_evaluation.py \
        --source_model models/stc_capsnet_grayscale.pth \
        --target_dataset ./data/ASVspoof2019_LA_eval \
        --spectrogram_type grayscale
else
    echo "ASVspoof dataset not found, skipping cross-dataset evaluation"
fi

# Generate visualizations
echo "Generating visualizations..."
python visualization.py --plot_type ablation
python visualization.py --plot_type training

echo "Training completed! Check models/, logs/, and results/ directories."


# scripts/evaluate_models.sh
#!/bin/bash

# Script to evaluate trained models

echo "Evaluating STC-CapsNet models..."

# Evaluate mel-spectrogram model
if [ -f "models/stc_capsnet_mel.pth" ]; then
    echo "Evaluating mel-spectrogram model..."
    python evaluate.py \
        --model_path models/stc_capsnet_mel.pth \
        --data_path ./data/for_processed \
        --spectrogram_type mel \
        --batch_size 32
else
    echo "Mel-spectrogram model not found"
fi

# Evaluate grayscale spectrogram model
if [ -f "models/stc_capsnet_grayscale.pth" ]; then
    echo "Evaluating grayscale spectrogram model..."
    python evaluate.py \
        --model_path models/stc_capsnet_grayscale.pth \
        --data_path ./data/for_processed \
        --spectrogram_type grayscale \
        --batch_size 32
else
    echo "Grayscale spectrogram model not found"
fi

# Benchmark models
echo "Benchmarking models..."
python benchmark.py \
    --data_path ./data/for_processed \
    --model_path models/stc_capsnet_mel.pth \
    --batch_size 32

echo "Evaluation completed!"


# scripts/run_inference.sh
#!/bin/bash

# Script to run inference on new audio files

if [ $# -eq 0 ]; then
    echo "Usage: $0 <audio_file_or_directory>"
    echo "Example: $0 test_audio.wav"
    echo "Example: $0 /path/to/audio/directory"
    exit 1
fi

INPUT_PATH=$1
MODEL_PATH="models/stc_capsnet_mel.pth"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found: $MODEL_PATH"
    echo "Please train the model first or provide correct model path"
    exit 1
fi

echo "Running inference on: $INPUT_PATH"

if [ -f "$INPUT_PATH" ]; then
    # Single file
    python inference.py \
        --model_path $MODEL_PATH \
        --audio_path "$INPUT_PATH" \
        --spectrogram_type mel \
        --output predictions_single.json
elif [ -d "$INPUT_PATH" ]; then
    # Directory
    python inference.py \
        --model_path $MODEL_PATH \
        --audio_dir "$INPUT_PATH" \
        --spectrogram_type mel \
        --output predictions_batch.json
else
    echo "Input path does not exist: $INPUT_PATH"
    exit 1
fi

echo "Inference completed!"


# docs/API.md
# STC-CapsNet API Documentation

## Model Classes

### STCCapsNet

Main model class implementing the Spatio-Temporal Convolutional Capsule Network.

```python
class STCCapsNet(nn.Module):
    def __init__(self, input_shape, num_classes=2):
        """
        Args:
            input_shape: Shape of input spectrograms (height, width)
            num_classes: Number of output classes (default: 2 for binary classification)
        """
```

**Methods:**
- `forward(x)`: Forward pass through the network
- `margin_loss(outputs, labels, m_pos=0.9, m_neg=0.1, lambda_val=0.5)`: Compute margin loss

### PrimaryCapsule

Primary capsule layer for encoding initial feature representations.

```python
class PrimaryCapsule(nn.Module):
    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output capsules
            dim_caps: Dimension of each capsule
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
        """
```

### RoutingCapsule

Higher-level capsule layer with dynamic routing.

```python
class RoutingCapsule(nn.Module):
    def __init__(self, in_caps, in_dim, out_caps, out_dim, num_routing=3):
        """
        Args:
            in_caps: Number of input capsules
            in_dim: Dimension of input capsules
            out_caps: Number of output capsules
            out_dim: Dimension of output capsules
            num_routing: Number of routing iterations
        """
```

## Dataset Classes

### AudioDeepfakeDataset

Main dataset class for loading and preprocessing audio files.

```python
class AudioDeepfakeDataset(Dataset):
    def __init__(self, data_path, split='train', spectrogram_type='mel', 
                 sample_rate=16000, n_fft=2048, hop_length=512, n_mels=128,
                 max_length=None, augment=False):
        """
        Args:
            data_path: Path to dataset directory
            split: Dataset split ('train', 'val', 'test')
            spectrogram_type: Type of spectrogram ('mel' or 'grayscale')
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            max_length: Maximum audio length in samples
            augment: Whether to apply data augmentation
        """
```

## Inference Class

### AudioDeepfakeInference

High-level inference class for audio deepfake detection.

```python
class AudioDeepfakeInference:
    def __init__(self, model_path, spectrogram_type='mel', device=None):
        """
        Args:
            model_path: Path to trained model file
            spectrogram_type: Type of spectrogram ('mel' or 'grayscale')
            device: PyTorch device (auto-detected if None)
        """
    
    def predict(self, audio_path):
        """
        Predict if audio is real or fake.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Prediction results with confidence scores
        """
    
    def predict_batch(self, audio_paths):
        """
        Predict for multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            list: List of prediction results
        """
```

## Utility Functions

### calculate_eer(y_true, y_scores)

Calculate Equal Error Rate (EER) for binary classification.

### setup_logging(log_dir='logs')

Setup logging configuration for training and evaluation.

### count_parameters(model)

Count the number of trainable parameters in a model.

## Command Line Interface

### Training

```bash
python main.py --data_path /path/to/data --spectrogram_type mel --batch_size 32 --epochs 100
```

### Evaluation

```bash
python evaluate.py --model_path model.pth --data_path /path/to/data --spectrogram_type mel
```

### Inference

```bash
# Single file
python inference.py --model_path model.pth --audio_path audio.wav

# Batch inference
python inference.py --model_path model.pth --audio_dir /path/to/audio/files
```

### Cross-dataset Evaluation

```bash
python cross_dataset_evaluation.py --source_model model.pth --target_dataset /path/to/asvspoof
```

## Configuration

### Config Class

Central configuration class for model and training parameters.

```python
class Config:
    # Data parameters
    SAMPLE_RATE = 16000
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    
    # Model parameters
    NUM_CLASSES = 2
    PRIMARY_CAPS_DIM = 8
    PRIMARY_CAPS_NUM = 32
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
```


# docs/USAGE.md
# STC-CapsNet Usage Guide

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended)
- At least 8GB RAM
- 10GB free disk space

### Install from Source

```bash
git clone https://github.com/CapsuleNetworks/STC-CapsNet.git
cd STC-CapsNet
pip install -r requirements.txt
pip install -e .
```

### Install with Conda

```bash
conda env create -f environment.yml
conda activate stc-capsnet
```

### Docker Installation

```bash
docker-compose up stc-capsnet-training
```

## Data Preparation

### FoR Dataset

1. Download the FoR dataset from the original authors
2. Extract the dataset to a directory
3. Prepare the dataset:

```bash
python data_preparation.py --source_path /path/to/for/dataset --output_path ./data/for_processed
```

### ASVspoof 2019 Dataset

1. Download ASVspoof 2019 LA dataset
2. Extract to `./data/ASVspoof2019_LA_eval/`

### Sample Dataset (for testing)

```bash
python data_preparation.py --create_sample --output_path ./data/sample --num_samples 1000
```

## Training

### Basic Training

```bash
# Train with mel-spectrograms
python main.py --data_path ./data/for_processed --spectrogram_type mel --batch_size 32 --epochs 100

# Train with grayscale spectrograms
python main.py --data_path ./data/for_processed --spectrogram_type grayscale --batch_size 32 --epochs 100
```

### Advanced Training Options

```bash
python main.py \
    --data_path ./data/for_processed \
    --spectrogram_type mel \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --save_model models/my_model.pth \
    --log_dir logs/experiment1
```

### Training with Custom Parameters

```python
from main import main
import argparse

# Create custom arguments
args = argparse.Namespace(
    data_path='./data/for_processed',
    spectrogram_type='mel',
    batch_size=64,
    epochs=150,
    lr=0.0005,
    save_model='models/custom_model.pth',
    log_dir='logs/custom_experiment'
)

# Run training
main(args)
```

## Evaluation

### Model Evaluation

```bash
python evaluate.py \
    --model_path models/stc_capsnet_mel.pth \
    --data_path ./data/for_processed \
    --spectrogram_type mel \
    --batch_size 32
```

### Cross-dataset Evaluation

```bash
python cross_dataset_evaluation.py \
    --source_model models/stc_capsnet_mel.pth \
    --target_dataset ./data/ASVspoof2019_LA_eval \
    --spectrogram_type mel
```

### Ablation Study

```bash
python train_ablation.py
```

## Inference

### Single File Inference

```bash
python inference.py \
    --model_path models/stc_capsnet_mel.pth \
    --audio_path test_audio.wav \
    --spectrogram_type mel
```

### Batch Inference

```bash
python inference.py \
    --model_path models/stc_capsnet_mel.pth \
    --audio_dir /path/to/audio/files \
    --spectrogram_type mel \
    --output batch_predictions.json
```

### Python API Inference

```python
from inference import AudioDeepfakeInference

# Initialize inference
detector = AudioDeepfakeInference(
    model_path='models/stc_capsnet_mel.pth',
    spectrogram_type='mel'
)

# Single prediction
result = detector.predict('test_audio.wav')
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")

# Batch prediction
audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
results = detector.predict_batch(audio_files)
```

## Visualization

### Generate Plots

```bash
# Plot spectrogram comparison
python visualization.py \
    --plot_type spectrograms \
    --real_audio real_sample.wav \
    --fake_audio fake_sample.wav

# Plot training curves
python visualization.py --plot_type training --log_file logs/training.log

# Plot ablation results
python visualization.py --plot_type ablation --results_file ablation_results.json
```

## Benchmarking

### Model Performance Benchmark

```bash
python benchmark.py \
    --model_path models/stc_capsnet_mel.pth \
    --data_path ./data/for_processed \
    --batch_size 32 \
    --iterations 100
```

### Compare Spectrogram Types

```bash
python benchmark.py --data_path ./data/sample
```

## Hyperparameter Tuning

```bash
python hyperparameter_tuning.py \
    --n_trials 50 \
    --data_path ./data/for_processed
```

## Model Export and Deployment

### Export to ONNX

```python
import torch
from model import STCCapsNet

# Load model
model = STCCapsNet(input_shape=(128, 128))
model.load_state_dict(torch.load('models/stc_capsnet_mel.pth'))
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 1, 128, 128)
torch.onnx.export(
    model,
    dummy_input,
    'models/stc_capsnet_mel.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

### TorchScript Export

```python
import torch
from model import STCCapsNet

# Load model
model = STCCapsNet(input_shape=(128, 128))
model.load_state_dict(torch.load('models/stc_capsnet_mel.pth'))
model.eval()

# Convert to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save('models/stc_capsnet_mel_scripted.pt')
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient accumulation
2. **Audio loading errors**: Ensure audio files are in supported formats (WAV, FLAC)
3. **Model convergence issues**: Adjust learning rate or use learning rate scheduling

### Performance Optimization

1. **Use mixed precision training**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(data)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

2. **Use DataLoader with multiple workers**:
```python
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True
)
```

### Memory Management

For large datasets or limited memory:

```python
# Enable gradient checkpointing
model = torch.utils.checkpoint.checkpoint_sequential(model, segments=2)

# Use smaller batch sizes with gradient accumulation
accumulation_steps = 4
for i, (data, labels) in enumerate(dataloader):
    outputs = model(data)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```