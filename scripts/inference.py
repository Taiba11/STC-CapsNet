"""
Single-File Inference for STC-CapsNet.

Usage:
    python scripts/inference.py \
        --checkpoint experiments/for_mel/best_model.pth \
        --audio_path path/to/audio.wav \
        --feature_type mel
"""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import STCCapsNet
from datasets.preprocessing import AudioPreprocessor, MelSpectrogramExtractor, GrayscaleSpectrogramExtractor


def main():
    parser = argparse.ArgumentParser(description="STC-CapsNet Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--feature_type", type=str, default="mel", choices=["mel", "grayscale"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = STCCapsNet(in_channels=1, num_classes=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Preprocess
    preprocessor = AudioPreprocessor(sample_rate=16000, duration=3.0)
    if args.feature_type == "mel":
        extractor = MelSpectrogramExtractor(sample_rate=16000)
    else:
        extractor = GrayscaleSpectrogramExtractor(sample_rate=16000)

    print(f"\nProcessing: {args.audio_path}")
    waveform = preprocessor.process(args.audio_path)
    spectrogram = extractor.extract(waveform).unsqueeze(0).to(device)

    with torch.no_grad():
        preds, confs = model.predict(spectrogram)

    pred = preds.item()
    c = confs.squeeze()
    label = "REAL (Bonafide)" if pred == 0 else "FAKE (Spoofed)"

    print(f"\n{'='*50}")
    print(f"  Feature:     {args.feature_type}")
    print(f"  Prediction:  {label}")
    print(f"  Confidence:  Real={c[0]:.4f} | Fake={c[1]:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
