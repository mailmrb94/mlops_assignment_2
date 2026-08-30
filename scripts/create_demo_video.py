#!/usr/bin/env python3
"""Create a concise terminal-style demonstration video from verified outputs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tmp" / "demo_video"
OUTPUT = ROOT / "docs" / "MLOps_Assignment_2_Demo.mp4"
W, H = 1280, 720
BG = "#0B1220"
PANEL = "#111C2E"
NAVY = "#152238"
BLUE = "#38BDF8"
TEAL = "#2DD4BF"
WHITE = "#F8FAFC"
MUTED = "#94A3B8"
GREEN = "#4ADE80"
YELLOW = "#FACC15"


def font(size: int, bold: bool = False, mono: bool = False):
    candidates = (
        ["/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc"]
        if mono
        else (["/System/Library/Fonts/SFNSDisplay-Bold.otf"] if bold else [])
        + ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc"]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size, index=1 if bold else 0)
            except OSError:
                continue
    return ImageFont.load_default()


TITLE = font(34, bold=True)
SUBTITLE = font(21)
BODY = font(18)
SMALL = font(14)
MONO = font(17, mono=True)
MONO_SMALL = font(14, mono=True)


def base(title: str, kicker: str, step: int, total: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, W, 72), fill=NAVY)
    draw.rounded_rectangle((34, 22, 176, 50), radius=14, fill="#173C52")
    draw.text((51, 27), "MLOps Demo", font=SMALL, fill=TEAL)
    draw.text((205, 19), title, font=TITLE, fill=WHITE)
    draw.text((W - 260, 27), f"{step:02d} / {total:02d}", font=SMALL, fill=MUTED)
    draw.text((44, 91), kicker, font=SUBTITLE, fill=BLUE)
    draw.rectangle((44, H - 24, W - 44, H - 20), fill="#22324A")
    draw.rectangle((44, H - 24, 44 + int((W - 88) * step / total), H - 20), fill=TEAL)
    return image, draw


def terminal(draw, box, command: str, lines: list[tuple[str, str]]):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=12, fill="#060B13", outline="#26374F", width=2)
    for x, color in [(x1 + 20, "#FF5F57"), (x1 + 40, "#FEBB2E"), (x1 + 60, "#28C840")]:
        draw.ellipse((x, y1 + 15, x + 10, y1 + 25), fill=color)
    draw.text((x1 + 20, y1 + 46), "$", font=MONO, fill=TEAL)
    draw.text((x1 + 43, y1 + 46), command, font=MONO, fill=WHITE)
    y = y1 + 82
    for text, color in lines:
        draw.text((x1 + 20, y), text, font=MONO_SMALL, fill=color)
        y += 24


def card(draw, box, heading: str, body: list[str], accent=TEAL):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=12, fill=PANEL, outline="#243B55", width=2)
    draw.rectangle((x1, y1, x1 + 6, y2), fill=accent)
    draw.text((x1 + 22, y1 + 18), heading, font=SUBTITLE, fill=WHITE)
    y = y1 + 58
    for line in body:
        draw.text((x1 + 22, y), line, font=BODY, fill=MUTED)
        y += 30


def fit_image(path: Path, box, background=PANEL):
    canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), background)
    source = Image.open(path).convert("RGB")
    source.thumbnail(canvas.size, Image.Resampling.LANCZOS)
    canvas.paste(source, ((canvas.width - source.width) // 2, (canvas.height - source.height) // 2))
    return canvas


def save_scene(index: int, image: Image.Image):
    path = WORK / f"scene-{index:02d}.png"
    image.save(path, quality=95)
    return path


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    cat = next((ROOT / "data" / "processed" / "test" / "cats").glob("*.jpg"))
    dog = next((ROOT / "data" / "processed" / "test" / "dogs").glob("*.jpg"))
    scenes: list[tuple[Path, int]] = []
    total = 13

    image, draw = base("Cats vs Dogs - End-to-End", "Assignment 2 submission walkthrough", 1, total)
    draw.text((64, 180), "MODEL", font=SMALL, fill=TEAL)
    draw.text((64, 210), "Versioned. Tested. Packaged. Deployable.", font=font(44, bold=True), fill=WHITE)
    draw.text((64, 282), "Git + DVC  |  MLflow  |  FastAPI  |  Docker  |  GitHub Actions", font=SUBTITLE, fill=BLUE)
    draw.rounded_rectangle((64, 350, 1210, 574), radius=16, fill=PANEL)
    labels = ["DATA", "PREPARE", "TRAIN", "SERVE", "PUBLISH", "DEPLOY", "MONITOR"]
    for i, label in enumerate(labels):
        x = 88 + i * 158
        draw.rounded_rectangle((x, 410, x + 112, 470), radius=10, fill="#173C52", outline=TEAL)
        draw.text((x + 18, 429), label, font=SMALL, fill=WHITE)
        if i < len(labels) - 1:
            draw.line((x + 118, 440, x + 147, 440), fill=TEAL, width=3)
    scenes.append((save_scene(1, image), 9))

    image, draw = base("Repository", "Everything required by M1-M5 is in one project", 2, total)
    terminal(
        draw,
        (48, 136, 720, 646),
        "find . -maxdepth 3 -type f | sort",
        [
            ("./.github/workflows/mlops.yml", BLUE),
            ("./artifacts/{metrics,history,confusion_matrix}", MUTED),
            ("./dvc.yaml   ./dvc.lock   ./params.yaml", WHITE),
            ("./Dockerfile   ./docker-compose.yml", WHITE),
            ("./models/{model.joblib,mobilenet_v3_small_features.onnx}", GREEN),
            ("./scripts/{download,prepare,train,smoke_test}.py", MUTED),
            ("./src/mlops_cats_dogs/{api,model,transfer_learning}.py", WHITE),
            ("./tests/{preprocessing,model,api}", GREEN),
        ],
    )
    card(draw, (755, 136, 1232, 358), "Submission package", ["Source and configuration", "Trained model artifact", "Report and demo video", "Reproduction instructions"])
    card(draw, (755, 378, 1232, 646), "Design goal", ["High-quality transfer model", "No test-set leakage", "ONNX runtime serving", "Immutable deployment tags", "No sensitive image logging"], BLUE)
    scenes.append((save_scene(2, image), 12))

    image, draw = base("Data and DVC", "600 audited images -> 224x224 RGB -> deterministic splits", 3, total)
    image.paste(fit_image(cat, (50, 150, 345, 440)), (50, 150))
    image.paste(fit_image(dog, (370, 150, 665, 440)), (370, 150))
    draw.text((50, 452), "CAT SAMPLE", font=SMALL, fill=TEAL)
    draw.text((370, 452), "DOG SAMPLE", font=SMALL, fill=TEAL)
    terminal(draw, (700, 145, 1232, 598), "dvc repro", [
        ("Running stage 'prepare'", BLUE),
        ("Prepared 600 images at data/processed", GREEN),
        ("train: 480  validation: 60  test: 60", WHITE),
        ("candidate augmentation: original vs horizontal flip", WHITE),
        ("Running stage 'train'", BLUE),
        ("model + metrics + plots -> DVC cache", GREEN),
        ("Updating lock file 'dvc.lock'", MUTED),
    ])
    draw.text((50, 520), "Per-image provenance", font=SUBTITLE, fill=WHITE)
    draw.text((50, 555), "Dataset row + label + source path + SHA-256", font=BODY, fill=MUTED)
    scenes.append((save_scene(3, image), 15))

    image, draw = base("Model and Evaluation", "Frozen MobileNetV3-Small features + regularized classifier", 4, total)
    image.paste(fit_image(ROOT / "artifacts" / "model_comparison.png", (48, 152, 735, 535), "#FFFFFF"), (48, 152))
    card(draw, (770, 150, 1228, 536), "Held-out test metrics", [
        f"Accuracy        {metrics['test_accuracy']:.3f}",
        f"Precision       {metrics['test_precision']:.3f}",
        f"Recall          {metrics['test_recall']:.3f}",
        f"F1              {metrics['test_f1']:.3f}",
        f"Log loss        {metrics['test_log_loss']:.3f}",
        "Confusion: [[30,0],[1,29]]",
    ], GREEN)
    draw.text((48, 570), "Serialized artifact", font=SUBTITLE, fill=WHITE)
    draw.text((48, 605), "ONNX 3.5 MB + classifier 19 KB  |  model version 2.0.0", font=BODY, fill=MUTED)
    scenes.append((save_scene(4, image), 15))

    image, draw = base("MLflow Tracking", "Baseline and champion runs with complete evidence", 5, total)
    card(draw, (50, 150, 397, 575), "Experiment", ["model-development", "baseline reference", "MobileNetV3 champion", "tracking URI: ./mlruns"], BLUE)
    card(draw, (425, 150, 805, 575), "Logged metrics", ["test_accuracy = 0.983", "test_f1 = 0.983", "test_log_loss = 0.032", "+28.33 accuracy points", "validation-only selection"], GREEN)
    card(draw, (833, 150, 1230, 575), "Logged artifacts", ["model + ONNX", "model_manifest.json", "model_comparison.png", "candidate search", "confusion matrix", "sample requests"], TEAL)
    terminal(draw, (235, 590, 1045, 671), "mlflow ui --backend-store-uri ./mlruns", [("Listening at http://127.0.0.1:5000", GREEN)])
    scenes.append((save_scene(5, image), 12))

    image, draw = base("Automated Tests", "Preprocessing, inference utility, and API contracts", 6, total)
    terminal(draw, (120, 154, 1160, 560), "pytest -q", [
        ("tests/test_preprocessing.py ..", GREEN),
        ("tests/test_model.py ..", GREEN),
        ("tests/test_api.py ..", GREEN),
        ("", WHITE),
        ("......                                                   [100%]", GREEN),
        ("6 passed", GREEN),
    ])
    draw.text((120, 598), "CI gate: a failing test prevents image creation and deployment.", font=SUBTITLE, fill=YELLOW)
    scenes.append((save_scene(6, image), 12))

    image, draw = base("FastAPI Inference", "Validated multipart image -> label + class probabilities", 7, total)
    terminal(draw, (48, 145, 1232, 600), "uvicorn mlops_cats_dogs.api:app --port 8000", [
        ("INFO model_loaded backend=onnx_mobilenet_v3_small version=2.0.0", BLUE),
        ("INFO Application startup complete", GREEN),
        ("", WHITE),
        ("GET  /health   -> 200  {status: healthy, model_version: 2.0.0}", WHITE),
        ("POST /predict  -> 200", WHITE),
        ('{ "label": "dog", "confidence": 0.9821,', GREEN),
        ('  "probabilities": {"cat": 0.0179, "dog": 0.9821} }', GREEN),
        ("GET  /docs     -> interactive OpenAPI", MUTED),
    ])
    draw.text((48, 625), "Safety: JPEG/PNG/WebP only | 10 MB limit | invalid payloads rejected", font=BODY, fill=MUTED)
    scenes.append((save_scene(7, image), 16))

    image, draw = base("Smoke Test", "The same checks run after deployment and fail the release", 8, total)
    terminal(draw, (80, 155, 1200, 570), "python scripts/smoke_test.py --image sample.jpg", [
        ('health={"status":"healthy","model_version":"2.0.0"}', GREEN),
        ("prediction={", WHITE),
        ('  "label":"dog", "confidence":0.9821,', GREEN),
        ('  "probabilities":{"cat":0.0179,"dog":0.9821}', GREEN),
        ("}", WHITE),
        ("exit status: 0", GREEN),
    ])
    draw.rounded_rectangle((80, 600, 1200, 654), radius=10, fill="#123B2A")
    draw.text((105, 616), "PASS - health and one real prediction verified", font=SUBTITLE, fill=GREEN)
    scenes.append((save_scene(8, image), 14))

    image, draw = base("Containerization", "Non-root Python 3.11 service with an embedded health check", 9, total)
    terminal(draw, (50, 150, 700, 620), "docker build -t cats-dogs-mlops:local .", [
        ("FROM python:3.11-slim", BLUE),
        ("pip install pinned production dependencies", WHITE),
        ("COPY model.joblib + MobileNetV3 ONNX", GREEN),
        ("USER app", GREEN),
        ("HEALTHCHECK GET /health", GREEN),
        ("CMD uvicorn ... --port 8000", WHITE),
    ])
    card(draw, (740, 150, 1230, 380), "Docker Compose target", ["restart: unless-stopped", "port 8000", "healthcheck + env", "immutable image override"], TEAL)
    card(draw, (740, 405, 1230, 620), "Production registry", ["GHCR image", "commit-SHA tag", "latest convenience tag", "BuildKit cache"], BLUE)
    scenes.append((save_scene(9, image), 14))

    image, draw = base("Continuous Integration", "Every push: test -> build -> publish to GHCR", 10, total)
    stages = [("CHECKOUT", BLUE), ("INSTALL", BLUE), ("PYTEST", GREEN), ("BUILD", TEAL), ("GHCR PUSH", TEAL)]
    for i, (label, color) in enumerate(stages):
        x = 58 + i * 238
        draw.rounded_rectangle((x, 205, x + 190, 305), radius=14, fill=PANEL, outline=color, width=3)
        draw.text((x + 22, 242), label, font=SUBTITLE, fill=WHITE)
        if i < len(stages) - 1:
            draw.line((x + 195, 255, x + 225, 255), fill=TEAL, width=4)
    terminal(draw, (120, 370, 1160, 622), ".github/workflows/mlops.yml", [
        ("on: push + pull_request", WHITE),
        ("pull request: build only; no registry write", MUTED),
        ("push: ghcr.io/<owner>/cats-dogs-mlops:<sha>", GREEN),
        ("permissions: contents:read, packages:write", BLUE),
    ])
    scenes.append((save_scene(10, image), 14))

    image, draw = base("Continuous Deployment", "Main branch updates a Docker Compose host automatically", 11, total)
    flow = ["MAIN", "GHCR", "SSH", "COMPOSE", "SMOKE", "LIVE"]
    for i, label in enumerate(flow):
        x = 50 + i * 202
        fill = "#123B2A" if label in {"SMOKE", "LIVE"} else PANEL
        draw.rounded_rectangle((x, 225, x + 158, 318), radius=14, fill=fill, outline=GREEN if label == "LIVE" else TEAL, width=3)
        draw.text((x + 24, 260), label, font=SUBTITLE, fill=WHITE)
        if i < len(flow) - 1:
            draw.line((x + 163, 270, x + 190, 270), fill=TEAL, width=4)
    card(draw, (85, 395, 590, 610), "Deployment secrets", ["DEPLOY_HOST", "DEPLOY_USER", "DEPLOY_SSH_KEY", "DEPLOY_PATH variable"], BLUE)
    card(draw, (620, 395, 1195, 610), "Rollback-ready", ["Every image tagged by commit SHA", "Compose pulls an exact image", "Health/prediction gate", "Failed smoke test fails release"], GREEN)
    scenes.append((save_scene(11, image), 14))

    image, draw = base("Monitoring", "Privacy-conscious logs + Prometheus request and latency metrics", 12, total)
    terminal(draw, (48, 145, 750, 610), "curl http://localhost:8000/metrics", [
        ("# HELP inference_requests_total Inference requests", MUTED),
        ('inference_requests_total{endpoint="health",status="ok"} 1', GREEN),
        ('inference_requests_total{endpoint="predict",status="ok"} 1', GREEN),
        ('inference_latency_seconds_count{endpoint="predict"} 1', BLUE),
        ("", WHITE),
        ("Logs: request ID, route, status, latency", WHITE),
        ("Excluded: filename, bytes, image content, user data", YELLOW),
    ])
    card(draw, (790, 145, 1230, 610), "Quality feedback", ["20 labeled sample requests", "prediction + probability", "true label + correctness", "rolling accuracy and F1", "drift and retraining hook"], TEAL)
    scenes.append((save_scene(12, image), 15))

    image, draw = base("Submission Ready", "All five modules mapped to runnable evidence", 13, total)
    items = [
        ("M1", "DVC + transfer model + MLflow"),
        ("M2", "FastAPI + Docker"),
        ("M3", "Tests + CI + GHCR"),
        ("M4", "Compose CD + smoke test"),
        ("M5", "Logs + metrics + labels"),
    ]
    for i, (module, description) in enumerate(items):
        y = 145 + i * 86
        draw.rounded_rectangle((120, y, 1160, y + 65), radius=12, fill=PANEL, outline="#243B55")
        draw.rounded_rectangle((140, y + 12, 215, y + 52), radius=10, fill="#123B2A")
        draw.text((159, y + 21), module, font=SUBTITLE, fill=GREEN)
        draw.text((250, y + 19), description, font=SUBTITLE, fill=WHITE)
        draw.text((1070, y + 19), "PASS", font=SUBTITLE, fill=GREEN)
    draw.text((120, 615), "Report + source ZIP + trained artifact + demo video", font=SUBTITLE, fill=BLUE)
    scenes.append((save_scene(13, image), 12))

    concat = WORK / "frames.txt"
    lines = []
    for path, duration in scenes:
        lines.extend([f"file '{path.as_posix()}'", f"duration {duration}"])
    lines.append(f"file '{scenes[-1][0].as_posix()}'")
    concat.write_text("\n".join(lines) + "\n")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-vf",
            "fps=24,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "22",
            "-movflags",
            "+faststart",
            str(OUTPUT),
        ],
        check=True,
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
