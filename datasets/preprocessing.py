"""
Preprocessing & Feature Extraction for STC-CapsNet (Section II-A).

Two feature extraction paths:
    1. Mel-Spectrograms (Eq. 1): Log-Mel(t,m) = log(S_mel(t,m) + ε)
    2. Grayscale Spectrograms (Eq. 2): S_gray(t,f) = normalize(|STFT|) * 255
"""

import numpy as np
import torch
import torchaudio
import librosa
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


class AudioPreprocessor:
    """
    Audio preprocessing pipeline (Section II-A).

    Steps:
        1. Spectral gating noise reduction
        2. Segmentation based on voice activity
        3. Silence removal (threshold-based)
        4. Resampling and normalization

    Args:
        sample_rate (int): Target sample rate.
        duration (float): Target duration in seconds.
        noise_reduction (bool): Apply spectral gating.
        silence_threshold_db (float): Silence removal threshold.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 3.0,
        noise_reduction: bool = True,
        silence_threshold_db: float = 30.0,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.noise_reduction = noise_reduction
        self.silence_threshold_db = silence_threshold_db

    def process(self, filepath: str) -> np.ndarray:
        """Load and preprocess an audio file."""
        waveform, sr = torchaudio.load(filepath)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0).numpy()

        # Resample
        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)

        # Noise reduction
        if self.noise_reduction:
            waveform = self._reduce_noise(waveform)

        # Silence removal
        waveform = self._remove_silence(waveform)

        # Normalize to [-1, 1]
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val

        # Pad or truncate
        waveform = self._fix_length(waveform)

        return waveform

    def _reduce_noise(self, waveform):
        try:
            import noisereduce as nr
            return nr.reduce_noise(y=waveform, sr=self.sample_rate)
        except ImportError:
            return waveform

    def _remove_silence(self, waveform):
        intervals = librosa.effects.split(waveform, top_db=self.silence_threshold_db)
        if len(intervals) > 0:
            waveform = np.concatenate([waveform[s:e] for s, e in intervals])
        return waveform

    def _fix_length(self, waveform):
        if len(waveform) > self.target_length:
            return waveform[:self.target_length]
        elif len(waveform) < self.target_length:
            return np.pad(waveform, (0, self.target_length - len(waveform)))
        return waveform


class MelSpectrogramExtractor:
    """
    Mel-Spectrogram Feature Extraction (Section II-A.1, Eq. 1).

    Log-Mel(t, m) = log(S_mel(t, m) + ε)

    Followed by mean-variance normalization.

    Args:
        sample_rate (int): Audio sample rate.
        n_fft (int): FFT size.
        hop_length (int): Hop length.
        n_mels (int): Number of mel filter banks.
        image_size (tuple): Output spatial size.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        image_size: tuple = (128, 128),
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.image_size = image_size

    def extract(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Extract log-mel spectrogram.

        Args:
            waveform: 1D numpy array.
        Returns:
            (1, H, W) tensor — single-channel mel spectrogram.
        """
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, window="hann",
        )

        # Log transform (Eq. 1)
        log_mel = np.log(mel_spec + 1e-9)

        # Mean-variance normalization
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

        # Resize to target size
        img = Image.fromarray(
            ((log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8) * 255).astype(np.uint8)
        )
        img = img.resize(self.image_size, Image.BILINEAR)

        # To tensor: (1, H, W)
        tensor = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0
        return tensor


class GrayscaleSpectrogramExtractor:
    """
    Grayscale Spectrogram Feature Extraction (Section II-A.2, Eq. 2).

    S_gray(t, f) = (S(t,f) - min(S)) / (max(S) - min(S)) * 255

    Args:
        sample_rate (int): Audio sample rate.
        n_fft (int): FFT size.
        hop_length (int): Hop length.
        image_size (tuple): Output spatial size.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        image_size: tuple = (128, 128),
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.image_size = image_size

    def extract(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Extract grayscale spectrogram.

        Args:
            waveform: 1D numpy array.
        Returns:
            (1, H, W) tensor — single-channel grayscale spectrogram.
        """
        # STFT magnitude
        stft = librosa.stft(y=waveform, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)

        # Normalize to [0, 255] (Eq. 2)
        s_min, s_max = magnitude.min(), magnitude.max()
        if s_max - s_min > 0:
            gray = (magnitude - s_min) / (s_max - s_min) * 255.0
        else:
            gray = np.zeros_like(magnitude)
        gray = gray.astype(np.uint8)

        # Resize
        img = Image.fromarray(gray)
        img = img.resize(self.image_size, Image.BILINEAR)

        # To tensor: (1, H, W)
        tensor = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0
        return tensor
