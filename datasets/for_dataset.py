"""
FoR (Fake or Real) Dataset Loader for STC-CapsNet (Section III-A).

Supports all four versions: for-original, for-norm, for-2seconds, for-rerecorded.
Combined dataset used for training (70% / 15% / 15% split).
"""

import os
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from .preprocessing import AudioPreprocessor, MelSpectrogramExtractor, GrayscaleSpectrogramExtractor


class FoRDataset(Dataset):
    """
    FoR dataset for STC-CapsNet.

    Args:
        file_list (list): List of (filepath, label) tuples.
        feature_type (str): "mel" or "grayscale".
        sample_rate (int): Audio sample rate.
        duration (float): Audio duration.
        augment (bool): Apply data augmentation.
    """

    def __init__(
        self,
        file_list: list,
        feature_type: str = "mel",
        sample_rate: int = 16000,
        duration: float = 3.0,
        augment: bool = False,
    ):
        self.file_list = file_list
        self.feature_type = feature_type
        self.augment = augment

        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate, duration=duration)

        if feature_type == "mel":
            self.extractor = MelSpectrogramExtractor(sample_rate=sample_rate)
        else:
            self.extractor = GrayscaleSpectrogramExtractor(sample_rate=sample_rate)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath, label = self.file_list[idx]

        # Preprocess audio
        waveform = self.preprocessor.process(filepath)

        # Data augmentation (Section III-B)
        if self.augment:
            waveform = self._augment(waveform)

        # Extract spectrogram
        spectrogram = self.extractor.extract(waveform)

        return spectrogram, torch.tensor(label, dtype=torch.long)

    def _augment(self, waveform):
        """Apply time-shifting and pitch-shifting augmentation."""
        if np.random.random() < 0.5:
            # Time shift
            shift = int(np.random.uniform(-0.1, 0.1) * len(waveform))
            waveform = np.roll(waveform, shift)

        if np.random.random() < 0.3:
            # Pitch shift
            import librosa
            n_steps = np.random.uniform(-2, 2)
            waveform = librosa.effects.pitch_shift(
                y=waveform, sr=self.preprocessor.sample_rate, n_steps=n_steps
            )

        return waveform


class FoRDatasetBuilder:
    """
    Builds train/val/test datasets from FoR directory.

    Args:
        data_dir (str): Root FoR directory.
        versions (list): FoR versions to include.
        train_ratio (float): Training split ratio.
        val_ratio (float): Validation split ratio.
        test_ratio (float): Test split ratio.
    """

    VERSIONS = ["for-original", "for-norm", "for-2seconds", "for-rerecorded"]
    AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}

    def __init__(
        self,
        data_dir: str,
        versions: Optional[List[str]] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        self.data_dir = Path(data_dir)
        self.versions = versions or self.VERSIONS
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def _collect_files(self):
        """Collect all audio files with labels."""
        all_files = []
        for version in self.versions:
            version_dir = self.data_dir / version
            if not version_dir.exists():
                version_dir = self.data_dir  # Try flat structure

            for label_name, label in [("real", 0), ("fake", 1)]:
                for search_dir in [version_dir, version_dir / "training",
                                   version_dir / "testing", version_dir / "validation"]:
                    label_dir = search_dir / label_name
                    if label_dir.exists():
                        for f in label_dir.rglob("*"):
                            if f.suffix.lower() in self.AUDIO_EXTENSIONS:
                                all_files.append((str(f), label))

        return all_files

    def build(self, feature_type="mel", augment_train=True, **kwargs):
        """
        Build train/val/test datasets.

        Returns:
            dict with 'train', 'val', 'test' FoRDataset instances.
        """
        all_files = self._collect_files()
        if not all_files:
            raise RuntimeError(f"No audio files found in {self.data_dir}")

        filepaths, labels = zip(*all_files)

        # Split: train / (val + test)
        train_fp, temp_fp, train_lb, temp_lb = train_test_split(
            filepaths, labels,
            test_size=(self.val_ratio + self.test_ratio),
            random_state=42, stratify=labels,
        )

        # Split: val / test
        relative_test = self.test_ratio / (self.val_ratio + self.test_ratio)
        val_fp, test_fp, val_lb, test_lb = train_test_split(
            temp_fp, temp_lb,
            test_size=relative_test,
            random_state=42, stratify=temp_lb,
        )

        return {
            "train": FoRDataset(
                list(zip(train_fp, train_lb)), feature_type=feature_type,
                augment=augment_train, **kwargs,
            ),
            "val": FoRDataset(
                list(zip(val_fp, val_lb)), feature_type=feature_type,
                augment=False, **kwargs,
            ),
            "test": FoRDataset(
                list(zip(test_fp, test_lb)), feature_type=feature_type,
                augment=False, **kwargs,
            ),
        }
