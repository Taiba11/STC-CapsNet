"""
ASVspoof 2019 Cross-Dataset Loader for STC-CapsNet.

Used for cross-dataset evaluation to test model generalization.
"""

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from .preprocessing import AudioPreprocessor, MelSpectrogramExtractor, GrayscaleSpectrogramExtractor


class ASVspoof2019Dataset(Dataset):
    """
    ASVspoof 2019 LA dataset for cross-dataset evaluation.

    Args:
        file_list (list): List of (filepath, label) tuples.
        feature_type (str): "mel" or "grayscale".
        sample_rate (int): Audio sample rate.
        duration (float): Audio duration.
    """

    def __init__(self, file_list, feature_type="mel", sample_rate=16000, duration=3.0):
        self.file_list = file_list
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate, duration=duration)
        if feature_type == "mel":
            self.extractor = MelSpectrogramExtractor(sample_rate=sample_rate)
        else:
            self.extractor = GrayscaleSpectrogramExtractor(sample_rate=sample_rate)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath, label = self.file_list[idx]
        waveform = self.preprocessor.process(filepath)
        spectrogram = self.extractor.extract(waveform)
        return spectrogram, torch.tensor(label, dtype=torch.long)


class ASVspoof2019Builder:
    """Build ASVspoof 2019 LA datasets from protocol files."""

    LABEL_MAP = {"bonafide": 0, "spoof": 1}

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def _parse_protocol(self, protocol_path, audio_dir):
        file_list = []
        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    audio_name = parts[1]
                    label_str = parts[4]
                    label = self.LABEL_MAP.get(label_str, 1)
                    fp = Path(audio_dir) / f"{audio_name}.flac"
                    if fp.exists():
                        file_list.append((str(fp), label))
        return file_list

    def get_eval_dataset(self, feature_type="mel", **kwargs):
        """Get evaluation set for cross-dataset testing."""
        protocol = self.data_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
        audio_dir = self.data_dir / "ASVspoof2019_LA_eval" / "flac"
        file_list = self._parse_protocol(str(protocol), str(audio_dir))
        return ASVspoof2019Dataset(file_list, feature_type=feature_type, **kwargs)
