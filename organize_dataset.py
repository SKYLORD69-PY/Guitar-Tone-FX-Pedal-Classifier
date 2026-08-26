"""
organize_dataset.py

Day 1 (dataset setup) -- Guitar Tone & FX Pedal Classifier
--------------------------------------------------------------------------
One-time helper that reorganizes a downloaded EGFxSet extraction into the
data/raw_audio/<class>/*.wav layout that extract_features.py expects.

EGFxSet ships as one folder per *pedal* -- Clean, BluesDriver, TubeScreamer,
RAT, Chorus, Flanger, Phaser, Digital-Delay, Sweep-Echo, TapeEcho,
Hall-Reverb, Plate-Reverb, Spring-Reverb -- each internally split further
into sub-folders per pickup configuration. This script doesn't care about
that internal nesting; it recursively collects every .wav under each
recognised effect folder and merges pedals into the broader tone classes
extract_features.py trains on:

    clean       <- Clean
    distortion  <- BluesDriver, TubeScreamer, RAT
    chorus      <- Chorus
    reverb      <- Hall-Reverb, Plate-Reverb, Spring-Reverb
    delay       <- Digital-Delay, Sweep-Echo, TapeEcho
    phaser      <- Phaser        (stretch class -- optional)
    flanger     <- Flanger       (stretch class -- optional)

Edit CLASS_MAP below if you'd rather keep pedals split out individually
(e.g. "blues_driver" / "tube_screamer" / "rat" as three separate classes
instead of one merged "distortion") or drop the phaser/flanger stretch
classes entirely.

Usage
-----
    python organize_dataset.py --source path/to/unzipped/EGFxSet --dest data/raw_audio
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Maps each raw EGFxSet folder name -> the class label it should land
# under. Matching is case/hyphen/space/underscore-insensitive (see
# _normalize) since Kaggle mirrors and the raw Zenodo zips haven't always
# agreed on exact casing (e.g. "Digital-Delay" vs "DigitalDelay").
CLASS_MAP: dict[str, str] = {
    "clean": "clean",
    "bluesdriver": "distortion",
    "tubescreamer": "distortion",
    "rat": "distortion",
    "chorus": "chorus",
    "phaser": "phaser",
    "flanger": "flanger",
    "digitaldelay": "delay",
    "sweepecho": "delay",
    "tapeecho": "delay",
    "hallreverb": "reverb",
    "platereverb": "reverb",
    "springreverb": "reverb",
}


def _normalize(name: str) -> str:
    """Collapse case/hyphen/space/underscore differences so 'Digital-Delay',
    'digital_delay', and 'DigitalDelay' all resolve to the same CLASS_MAP key."""
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def organize(source: Path, dest: Path, use_symlinks: bool = True) -> dict[str, int]:
    """Recursively find every .wav under each recognised effect folder in
    `source` and place it under dest/<mapped_class>/, regardless of how
    deeply nested it is inside that effect folder.

    Returns a dict of {class_label: clip_count} for a quick summary.
    """
    if not source.exists():
        raise FileNotFoundError(f"--source does not exist: {source}")

    effect_dirs = [d for d in source.iterdir() if d.is_dir()]
    if not effect_dirs:
        raise FileNotFoundError(
            f"No sub-folders found under {source} -- point --source at the "
            f"folder that directly contains Clean/, Chorus/, BluesDriver/, etc."
        )

    counts: dict[str, int] = {}
    unmatched: list[str] = []

    for effect_dir in sorted(effect_dirs):
        label = CLASS_MAP.get(_normalize(effect_dir.name))
        if label is None:
            unmatched.append(effect_dir.name)
            continue

        target_dir = dest / label
        target_dir.mkdir(parents=True, exist_ok=True)

        wav_files = sorted(effect_dir.rglob("*.wav"))
        for i, wav_path in enumerate(wav_files):
            # Prefix with the source pedal's name so files from multiple
            # pedals merged into one class (e.g. BluesDriver + RAT ->
            # distortion) never collide on filename.
            target_path = target_dir / f"{effect_dir.name.lower()}_{i:04d}.wav"
            if target_path.exists() or target_path.is_symlink():
                continue
            if use_symlinks:
                try:
                    target_path.symlink_to(wav_path.resolve())
                except OSError:
                    shutil.copy2(wav_path, target_path)  # e.g. symlinks unsupported here
            else:
                shutil.copy2(wav_path, target_path)

        counts[label] = counts.get(label, 0) + len(wav_files)
        logger.info(
            "%-14s (%d clips) -> data/raw_audio/%s/", effect_dir.name, len(wav_files), label
        )

    if unmatched:
        logger.warning("Skipped unrecognized folders (not in CLASS_MAP): %s", unmatched)

    logger.info("Done. Class totals: %s", counts)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorganize a downloaded EGFxSet folder into the data/raw_audio/<class>/ layout."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the unzipped EGFxSet root (the folder directly containing Clean/, Chorus/, etc.)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/raw_audio"),
        help="Where to build the class-labeled folders (default: data/raw_audio)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy audio files instead of symlinking. Uses much more disk but is portable "
        "across drives/filesystems that don't support symlinks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    organize(args.source, args.dest, use_symlinks=not args.copy)


if __name__ == "__main__":
    main()
