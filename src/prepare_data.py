"""Turn Kaggle's six fruit-condition folders into two classes.

Preserve Kaggle's original test split. Divide its training images into
training and validation (80/20). Keep fruit/condition representation in each.
"""
import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path
from PIL import Image, UnidentifiedImageError

from src.config import RAW, PROCESSED, RESULTS, SEED

FOLDERS = {
    "freshapples": "fresh",
    "freshbanana": "fresh",
    "freshoranges": "fresh",
    "rottenapples": "rotten",
    "rottenbanana": "rotten",
    "rottenoranges": "rotten",
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def locate_source(root: Path):
    """Find a directory whose train/ and test/ each contain the six classes."""
    for candidate in [root] + sorted(p for p in root.rglob("*") if p.is_dir()):
        train, test = candidate / "train", candidate / "test"
        if train.is_dir() and test.is_dir():
            def names(folder):
                return {p.name.lower() for p in folder.iterdir() if p.is_dir()}
            if set(FOLDERS).issubset(names(train)) and set(FOLDERS).issubset(names(test)):
                return train, test
    raise FileNotFoundError(
        "Could not find train/ and test/ with freshapples, freshbanana, "
        "freshoranges, rottenapples, rottenbanana, rottenoranges. "
        f"Extract the Kaggle dataset somewhere inside: {root}"
    )


def get_folders(base: Path):
    return {p.name.lower(): p for p in base.iterdir() if p.is_dir()}


def collect_valid(folder: Path):
    valid = []
    skipped = 0
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in EXTENSIONS:
            continue
        try:
            with Image.open(p) as img:
                img.verify()
            valid.append(p)
        except (OSError, ValueError, UnidentifiedImageError):
            skipped += 1
    return valid, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW, help="Directory containing extracted Kaggle data")
    parser.add_argument("--validation-ratio", type=float, default=0.2,
                        help="Proportion of Kaggle training images reserved for validation")
    args = parser.parse_args()
    if not 0 < args.validation_ratio < 1:
        parser.error("--validation-ratio must be between 0 and 1")
    train_root, test_root = locate_source(args.raw)
    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)
    for split in ("train", "val", "test"):
        for label in ("fresh", "rotten"):
            (PROCESSED / split / label).mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    rows = []
    skipped_total = 0
    for original, label in FOLDERS.items():
        train_files, skipped = collect_valid(get_folders(train_root)[original])
        test_files, skipped_test = collect_valid(get_folders(test_root)[original])
        skipped_total += skipped + skipped_test
        if len(train_files) < 2 or not test_files:
            raise ValueError(f"Not enough valid images for {original}")
        rng.shuffle(train_files)
        n_val = max(1, round(len(train_files) * args.validation_ratio))
        for split, files in (("train", train_files[n_val:]),
                             ("val", train_files[:n_val]),
                             ("test", test_files)):
            for source in files:
                # Prefix the original class name to avoid filename collisions.
                destination = PROCESSED / split / label / f"{original}_{source.name}"
                if destination.exists():
                    raise FileExistsError(f"Duplicate destination name: {destination}")
                shutil.copy2(source, destination)
            rows.append({"original_category": original, "label": label,
                         "split": split, "images": len(files)})

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "dataset_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["original_category", "label", "split", "images"])
        writer.writeheader()
        writer.writerows(rows)
    print("Dataset ready:", PROCESSED)
    for split in ("train", "val", "test"):
        counts = Counter({label: sum(r["images"] for r in rows
                                 if r["split"] == split and r["label"] == label)
                          for label in ("fresh", "rotten")})
        print(f"{split}: {dict(counts)}")
    print("Corrupted images skipped:", skipped_total)
    print("Saved counts:", RESULTS / "dataset_counts.csv")
    print("Note: inspect near-duplicate images across splits if working toward publication-quality results.")


if __name__ == "__main__":
    main()
