#!/usr/bin/env python3
"""Download a reproducible subset of the Microsoft/Kaggle Cats-vs-Dogs data.

The original Google teaching archive now blocks automated requests. This downloader
uses the public Hugging Face dataset-server mirror of the same Microsoft dataset and
records row-level provenance. It fetches only the requested number per class.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "microsoft/cats_vs_dogs"
API_URL = "https://datasets-server.huggingface.co/rows"
# In this published split, valid images are ordered as cats followed by dogs.
CLASS_OFFSETS = {"cats": 0, "dogs": 11741}


def fetch_json(url: str) -> dict:
    for attempt in range(6):
        request = urllib.request.Request(url, headers={"User-Agent": "mlops-assignment/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "mlops-assignment/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--per-class", type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = []
    for expected_label, (class_name, start_offset) in enumerate(CLASS_OFFSETS.items()):
        class_dir = args.output / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(class_dir.glob(f"{class_name[:-1]}.*.jpg"))
        if len(existing) >= args.per_class:
            for output_file in existing[: args.per_class]:
                row_index = int(output_file.stem.split(".")[-1])
                provenance.append(
                    {
                        "dataset": DATASET,
                        "row": row_index,
                        "label": class_name[:-1],
                        "file": f"{class_name}/{output_file.name}",
                        "sha256": hashlib.sha256(output_file.read_bytes()).hexdigest(),
                    }
                )
            print(f"{class_name}: resumed {args.per_class}/{args.per_class}")
            continue
        downloaded = 0
        while downloaded < args.per_class:
            length = min(100, args.per_class - downloaded)
            query = urllib.parse.urlencode(
                {
                    "dataset": DATASET,
                    "config": "default",
                    "split": "train",
                    "offset": start_offset + downloaded,
                    "length": length,
                }
            )
            payload = fetch_json(f"{API_URL}?{query}")
            for item in payload["rows"]:
                label = int(item["row"]["labels"])
                if label != expected_label:
                    raise RuntimeError(
                        f"dataset ordering changed at row {item['row_idx']}: expected label "
                        f"{expected_label}, got {label}"
                    )
                image_url = item["row"]["image"]["src"]
                filename = f"{class_name[:-1]}.{item['row_idx']}.jpg"
                output_file = class_dir / filename
                if output_file.exists():
                    image_bytes = output_file.read_bytes()
                else:
                    image_bytes = fetch_bytes(image_url)
                    output_file.write_bytes(image_bytes)
                digest = hashlib.sha256(image_bytes).hexdigest()
                provenance.append(
                    {
                        "dataset": DATASET,
                        "row": item["row_idx"],
                        "label": class_name[:-1],
                        "file": f"{class_name}/{filename}",
                        "sha256": digest,
                    }
                )
            downloaded += len(payload["rows"])
            print(f"{class_name}: {downloaded}/{args.per_class}")
            time.sleep(0.05)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output / "SOURCE.txt").write_text(
        "dataset=microsoft/cats_vs_dogs\n"
        "origin=Microsoft Research Asirra / Kaggle Dogs vs Cats\n"
        "mirror=https://huggingface.co/datasets/microsoft/cats_vs_dogs\n"
        "version=TensorFlow Datasets 4.0.1 compatible cleaned set\n"
    )
    print(f"Dataset ready at {args.output}; {len(provenance)} images with SHA-256 provenance")


if __name__ == "__main__":
    main()
