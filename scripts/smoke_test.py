#!/usr/bin/env python3
"""Fail-fast post-deployment health and prediction checks."""

from __future__ import annotations

import argparse
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(url: str, data: bytes | None = None, headers: dict[str, str] | None = None) -> str:
    with urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=headers or {}), timeout=10
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned {response.status}")
        return response.read().decode()


def multipart(image: Path) -> tuple[bytes, str]:
    boundary = "----mlops-smoke-boundary"
    content_type = mimetypes.guess_type(image.name)[0] or "image/jpeg"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + image.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=12)
    args = parser.parse_args()
    health_error = None
    for _ in range(args.retries):
        try:
            health = request(f"{args.base_url}/health")
            print(f"health={health}")
            break
        except (OSError, urllib.error.URLError) as exc:
            health_error = exc
            time.sleep(2)
    else:
        raise SystemExit(f"health check failed: {health_error}")
    body, content_type = multipart(args.image)
    prediction = request(
        f"{args.base_url}/predict", body, {"Content-Type": content_type}
    )
    if '"label"' not in prediction or '"probabilities"' not in prediction:
        raise SystemExit(f"invalid prediction response: {prediction}")
    print(f"prediction={prediction}")


if __name__ == "__main__":
    main()

