"""
Batch Spectrogram Generation for STC-CapsNet.

Usage:
    python scripts/generate_spectrograms.py \
        --data_dir data/FoR --output_dir data/spectrograms/for_mel --feature_type mel
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.preprocessing import AudioPreprocessor, MelSpectrogramExtractor, GrayscaleSpectrogramExtractor
from PIL import Image
import numpy as np


def process_file(args_tuple):
    audio_path, output_path, feature_type, sr, duration = args_tuple
    try:
        preprocessor = AudioPreprocessor(sample_rate=sr, duration=duration)
        waveform = preprocessor.process(audio_path)

        if feature_type == "mel":
            extractor = MelSpectrogramExtractor(sample_rate=sr)
        else:
            extractor = GrayscaleSpectrogramExtractor(sample_rate=sr)

        tensor = extractor.extract(waveform)
        img_array = (tensor.squeeze(0).numpy() * 255).astype(np.uint8)
        Image.fromarray(img_array).save(output_path)
        return True, audio_path
    except Exception as e:
        return False, f"{audio_path}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Generate spectrograms")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--feature_type", type=str, default="mel", choices=["mel", "grayscale"])
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files
    tasks = []
    data_dir = Path(args.data_dir)
    for f in data_dir.rglob("*"):
        if f.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
            rel = f.relative_to(data_dir)
            out = str(output_dir / rel.with_suffix(".png"))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            tasks.append((str(f), out, args.feature_type, 16000, 3.0))

    print(f"Processing {len(tasks)} files → {output_dir}")

    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(process_file, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks)):
            success, info = fut.result()
            if success:
                ok += 1
            else:
                fail += 1

    print(f"Done! Success: {ok}, Failed: {fail}")


if __name__ == "__main__":
    main()
