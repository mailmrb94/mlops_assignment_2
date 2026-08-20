#!/usr/bin/env python3
"""CLI wrapper for model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops_cats_dogs.train import train_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.params.read_text())["train"]
    metrics = train_model(args.data, args.model, args.artifacts, config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

