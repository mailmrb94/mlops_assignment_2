#!/usr/bin/env python3
"""Create deterministic 80/10/10 splits of standardized 224x224 RGB images."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops_cats_dogs.preprocessing import preprocess_image  # noqa: E402


def discover(root: Path, label: str) -> list[Path]:
    matches = []
    for singular_or_plural in (label, f"{label}s"):
        matches.extend(root.rglob(f"{singular_or_plural}/*.jpg"))
    return sorted(set(matches))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.params.read_text())["prepare"]
    rng = np.random.default_rng(int(config["seed"]))
    max_per_class = int(config["max_per_class"])
    train_fraction = float(config["train_fraction"])
    validation_fraction = float(config["validation_fraction"])
    image_size = int(config["image_size"])

    if args.output.exists():
        shutil.rmtree(args.output)
    manifest_rows = []
    for label in ("cat", "dog"):
        candidates = discover(args.input, label)
        if len(candidates) < max_per_class:
            raise ValueError(f"need {max_per_class} {label} images; found {len(candidates)}")
        selected = [candidates[index] for index in rng.permutation(len(candidates))[:max_per_class]]
        train_end = round(max_per_class * train_fraction)
        validation_end = train_end + round(max_per_class * validation_fraction)
        assignments = (
            [("train", path) for path in selected[:train_end]]
            + [("validation", path) for path in selected[train_end:validation_end]]
            + [("test", path) for path in selected[validation_end:]]
        )
        for split, source in assignments:
            output = args.output / split / f"{label}s" / source.name
            output.parent.mkdir(parents=True, exist_ok=True)
            preprocess_image(source, (image_size, image_size)).save(
                output, format="JPEG", quality=90, optimize=True
            )
            manifest_rows.append([str(output.relative_to(args.output)), split, label, str(source)])

    with (args.output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["processed_path", "split", "label", "source_path"])
        writer.writerows(sorted(manifest_rows))
    print(f"Prepared {len(manifest_rows)} images at {args.output}")


if __name__ == "__main__":
    main()

