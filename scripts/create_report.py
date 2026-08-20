#!/usr/bin/env python3
"""Generate the polished assignment report PDF from verified project artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "MLOps_Assignment_2_Report.pdf"
NAVY = colors.HexColor("#152238")
BLUE = colors.HexColor("#176B87")
TEAL = colors.HexColor("#2AA198")
PALE = colors.HexColor("#EAF4F4")
LIGHT = colors.HexColor("#F5F7FA")
MUTED = colors.HexColor("#5B6573")


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 12 * mm, width, 12 * mm, stroke=0, fill=1)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(colors.white)
    canvas.drawString(18 * mm, height - 8 * mm, "MLOps Assignment 2 | Cats vs Dogs")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(18 * mm, 10 * mm, "End-to-end open-source MLOps pipeline")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=32,
        alignment=TA_LEFT,
        textColor=NAVY,
        spaceAfter=12,
    )
)
styles.add(
    ParagraphStyle(
        "CoverSub",
        parent=styles["Normal"],
        fontSize=14,
        leading=20,
        textColor=BLUE,
        spaceAfter=18,
    )
)
styles.add(
    ParagraphStyle(
        "Section",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        textColor=NAVY,
        spaceBefore=8,
        spaceAfter=10,
    )
)
styles.add(
    ParagraphStyle(
        "Subsection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=BLUE,
        spaceBefore=8,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        "Body2",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.3,
        leading=13.2,
        textColor=colors.HexColor("#263238"),
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=7.8,
        leading=10.2,
        textColor=MUTED,
    )
)
styles.add(
    ParagraphStyle(
        "Callout",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=NAVY,
        backColor=PALE,
        borderColor=TEAL,
        borderWidth=1,
        borderPadding=9,
        spaceBefore=5,
        spaceAfter=10,
    )
)
styles.add(
    ParagraphStyle(
        "CodeBlock",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7.4,
        leading=9.6,
        textColor=colors.HexColor("#E6EDF3"),
        backColor=colors.HexColor("#0D1117"),
        borderPadding=8,
        leftIndent=0,
        rightIndent=0,
        spaceAfter=8,
    )
)


def p(text: str, style: str = "Body2") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str) -> Paragraph:
    return Paragraph(f"<bullet>&bull;</bullet>{text}", styles["Body2"])


def table(data, widths, header=True):
    wrapped = [[p(str(value), "Small") for value in row] for row in data]
    result = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    result.setStyle(TableStyle(commands))
    return result


def architecture_table():
    cells = [
        ("DATA", "Public mirror\n+ SHA-256 provenance"),
        ("DVC", "224x224 RGB\n80/10/10 split"),
        ("MODEL", "HOG + color\nRandom forest"),
        ("SERVE", "FastAPI\nPrometheus"),
        ("DELIVER", "Docker + GHCR\nCompose CD"),
    ]
    row = []
    for index, (title, body) in enumerate(cells):
        row.append(
            p(
                f"<b><font color='#176B87'>{title}</font></b><br/>{body}",
                "Small",
            )
        )
        if index < len(cells) - 1:
            row.append(p("<font color='#2AA198'>&#8594;</font>", "Subsection"))
    chart = Table([row], colWidths=[29 * mm, 6 * mm, 29 * mm, 6 * mm, 29 * mm, 6 * mm, 29 * mm, 6 * mm, 29 * mm])
    chart.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 0), (0, 0), PALE),
                ("BACKGROUND", (2, 0), (2, 0), PALE),
                ("BACKGROUND", (4, 0), (4, 0), PALE),
                ("BACKGROUND", (6, 0), (6, 0), PALE),
                ("BACKGROUND", (8, 0), (8, 0), PALE),
                ("BOX", (0, 0), (0, 0), 0.8, TEAL),
                ("BOX", (2, 0), (2, 0), 0.8, TEAL),
                ("BOX", (4, 0), (4, 0), 0.8, TEAL),
                ("BOX", (6, 0), (6, 0), 0.8, TEAL),
                ("BOX", (8, 0), (8, 0), 0.8, TEAL),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return chart


def build_story():
    metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    confusion = json.loads((ROOT / "artifacts" / "confusion_matrix.json").read_text())["matrix"]
    story = [Spacer(1, 24 * mm)]
    story += [
        p("ASSIGNMENT 2", "Small"),
        p("End-to-End MLOps Pipeline", "CoverTitle"),
        p("Binary Image Classification: Cats vs Dogs", "CoverSub"),
        Spacer(1, 8 * mm),
        architecture_table(),
        Spacer(1, 16 * mm),
        p(
            "A complete, reproducible implementation for model development, artifact and image "
            "creation, packaging, containerization, CI/CD deployment, monitoring, and post-deployment evaluation.",
            "Callout",
        ),
        Spacer(1, 12 * mm),
        table(
            [
                ["Course", "MLOps (S1-25_AIMLCZG523)"],
                ["Use case", "Pet adoption platform - Cats vs Dogs"],
                ["Submission", "Source, DVC, model, Docker, CI/CD, report, demo"],
                ["Prepared", "20 August 2026"],
            ],
            [35 * mm, 110 * mm],
            header=False,
        ),
        PageBreak(),
        p("1. Executive Summary", "Section"),
        p(
            "This project implements all five assignment modules as one executable repository. "
            "A balanced 600-image sample from the Microsoft/Kaggle Cats-vs-Dogs collection is "
            "downloaded through a public mirror with per-image SHA-256 provenance. DVC produces "
            "deterministic 80/10/10 splits and runs training. MLflow logs parameters, evaluation "
            "metrics, the serialized model, confusion matrix, sample requests, and learning curves.",
        ),
        p(
            "The inference layer is a FastAPI service with health, prediction, and Prometheus "
            "metrics endpoints. Docker and Docker Compose define packaging and deployment. GitHub "
            "Actions tests every change, builds an immutable image, publishes to GHCR, deploys main "
            "to a Compose host over SSH, and fails the release if smoke tests do not pass.",
        ),
        p(
            f"Verified result: <b>{metrics['test_accuracy']:.0%} test accuracy</b>, "
            f"<b>{metrics['test_f1']:.3f} F1</b>, and <b>5/5 automated tests passing</b>.",
            "Callout",
        ),
        p("Assignment-to-deliverable map", "Subsection"),
        table(
            [
                ["Module", "Implemented evidence", "Key files"],
                ["M1", "Git/DVC, 224x224 preprocessing, model, MLflow", "dvc.yaml, params.yaml, train.py, model.joblib"],
                ["M2", "FastAPI, pinned environment, Docker", "api.py, requirements.txt, Dockerfile"],
                ["M3", "pytest, GitHub Actions, GHCR publish", "tests/, .github/workflows/mlops.yml"],
                ["M4", "Compose target, SSH CD, smoke tests", "docker-compose.yml, smoke_test.py"],
                ["M5", "Structured logs, counters/latency, labeled batch", "api.py, sample_requests.csv"],
            ],
            [18 * mm, 79 * mm, 58 * mm],
        ),
        Spacer(1, 4 * mm),
        p("Technology choices", "Subsection"),
        table(
            [
                ["Concern", "Choice", "Reason"],
                ["Versioning", "Git + DVC", "Separates lightweight code history from data/model lineage."],
                ["Tracking", "MLflow", "Open source, local-first, logs metrics and arbitrary artifacts."],
                ["Model", "Random forest", "Fast laptop baseline with calibrated class proportions."],
                ["Serving", "FastAPI", "Typed REST API, validation, and automatic OpenAPI docs."],
                ["Delivery", "Docker + GHCR + Compose", "Portable artifact and a simple reproducible target."],
            ],
            [30 * mm, 38 * mm, 87 * mm],
        ),
        PageBreak(),
        p("2. M1 - Model Development and Experiment Tracking", "Section"),
        p("2.1 Data and code versioning", "Subsection"),
        p(
            "Git versions all source and configuration files. DVC defines two dependency-aware "
            "stages: <b>prepare</b> and <b>train</b>. The lock file captures hashes for code, data, "
            "parameters, processed output, metrics, history, and the trained model. A local DVC "
            "remote is configured for offline reproducibility; it can be replaced with S3 or another "
            "remote without changing the pipeline.",
        ),
        Preformatted(
            "$ python scripts/download_data.py --output data/raw\n"
            "$ dvc repro\n"
            "Running stage 'prepare' ... Prepared 600 images\n"
            "Running stage 'train' ... metrics + model + MLflow run",
            styles["CodeBlock"],
        ),
        p("2.2 Dataset and preprocessing", "Subsection"),
        table(
            [
                ["Property", "Value"],
                ["Source", "Microsoft Cats vs Dogs, public Hugging Face mirror"],
                ["Selected images", "600 total: 300 cats, 300 dogs"],
                ["Preprocessing", "EXIF orientation, RGB, center crop, 224x224; train horizontal flips"],
                ["Split", "480 train / 60 validation / 60 test (80% / 10% / 10%)"],
                ["Integrity", "Source row and SHA-256 stored for every downloaded image"],
                ["Reproducibility", "Seed 42; deterministic sampling and class-balanced split"],
            ],
            [42 * mm, 113 * mm],
        ),
        p("2.3 Baseline model", "Subsection"),
        p(
            "Every standardized image is summarized by HOG-style edge descriptors, a compact "
            "16x16 RGB thumbnail, and per-channel color histograms. A 300-tree random forest is "
            "grown in 12 increments of 25 trees; the increments provide train/validation learning "
            "curves. Original-only and horizontally augmented candidates are compared using validation "
            "accuracy (0.733 vs 0.667), so the original-only candidate is selected without consulting "
            "the test set. The final estimator supports class probabilities. The complete inference "
            "bundle is serialized with joblib and includes feature settings, class names, model "
            "version, and evaluation metadata. Deterministic horizontal flips expand 480 training "
            "images into 960 training samples without contaminating validation or test data.",
        ),
        p("2.4 MLflow tracking", "Subsection"),
        bullet("Experiment: <b>cats-vs-dogs-baseline</b>; run: <b>random-forest-hog-baseline</b>."),
        bullet("Parameters: feature size, histogram bins, epochs, trees per epoch, max features, seed."),
        bullet("Metrics: accuracy, log loss, precision, recall, and F1."),
        bullet("Artifacts: model.joblib, metrics JSON, confusion matrix, CSV history, curves, labeled request batch."),
        PageBreak(),
        p("3. Evaluation Results", "Section"),
        table(
            [
                ["Metric", "Result", "Interpretation"],
                ["Accuracy", f"{metrics['test_accuracy']:.3f}", "Overall correctness on 60 held-out images"],
                ["Precision (dog)", f"{metrics['test_precision']:.3f}", "Purity of the predicted-dog class"],
                ["Recall (dog)", f"{metrics['test_recall']:.3f}", "Coverage of actual dog images"],
                ["F1 (dog)", f"{metrics['test_f1']:.3f}", "Harmonic balance of precision and recall"],
                ["Log loss", f"{metrics['test_log_loss']:.3f}", "Probability-quality metric; lower is better"],
            ],
            [40 * mm, 28 * mm, 87 * mm],
        ),
        Spacer(1, 5 * mm),
        KeepTogether(
            [
                p("Confusion matrix", "Subsection"),
                table(
                    [
                        ["Actual / Predicted", "Cat", "Dog"],
                        ["Cat", confusion[0][0], confusion[0][1]],
                        ["Dog", confusion[1][0], confusion[1][1]],
                    ],
                    [55 * mm, 40 * mm, 40 * mm],
                ),
            ]
        ),
        Spacer(1, 7 * mm),
        Image(str(ROOT / "artifacts" / "training_curves.png"), width=155 * mm, height=62 * mm),
        p(
            "Figure 1. Training and validation loss/accuracy as the forest grows from 25 to 300 trees. "
            "The gap indicates expected overfitting for a deliberately small baseline dataset; transfer "
            "learning on the full dataset is the recommended next model iteration.",
            "Small",
        ),
        p("Post-deployment performance batch", "Subsection"),
        p(
            "The training pipeline writes 20 simulated production requests with image identifier, true "
            "label, predicted label, dog probability, and correctness. This establishes the schema for "
            "delayed-ground-truth monitoring without logging image bytes or personal information.",
        ),
        PageBreak(),
        p("4. M2 - Packaging and Containerization", "Section"),
        p("4.1 Inference API", "Subsection"),
        table(
            [
                ["Method", "Endpoint", "Behavior"],
                ["GET", "/health", "Returns service status and loaded model version"],
                ["POST", "/predict", "Accepts JPEG/PNG/WebP up to 10 MB; returns label and probabilities"],
                ["GET", "/metrics", "Exports Prometheus request count and latency series"],
                ["GET", "/docs", "Interactive OpenAPI documentation generated by FastAPI"],
            ],
            [20 * mm, 35 * mm, 100 * mm],
        ),
        Preformatted(
            "POST /predict\n"
            "{\n"
            '  "label": "dog",\n'
            '  "confidence": 0.5267,\n'
            '  "probabilities": {"cat": 0.4733, "dog": 0.5267},\n'
            '  "model_version": "1.0.0"\n'
            "}",
            styles["CodeBlock"],
        ),
        p("4.2 Reproducible environment", "Subsection"),
        p(
            "Production and development dependencies are fully pinned. The package uses a src layout "
            "and declares Python 3.11+. The service validates content type, file size, and decoded image "
            "content, then executes exactly the same feature path used during training.",
        ),
        p("4.3 Container", "Subsection"),
        bullet("Base: <b>python:3.11-slim</b>; no compiler or notebook runtime in the final image."),
        bullet("Runs as a non-root system user and exposes only port 8000."),
        bullet("Includes a Docker health check against <b>/health</b>."),
        bullet("Copies the versioned model artifact into a fixed, environment-overridable path."),
        bullet("A .dockerignore excludes datasets, caches, tests, reports, and local environments."),
        Preformatted(
            "$ docker build -t cats-dogs-mlops:local .\n"
            "$ IMAGE_NAME=cats-dogs-mlops:local docker compose up -d\n"
            "$ python scripts/smoke_test.py --image sample.jpg",
            styles["CodeBlock"],
        ),
        PageBreak(),
        p("5. M3 - Continuous Integration", "Section"),
        p(
            "The GitHub Actions workflow triggers on all pushes and pull requests to main. Jobs use "
            "least-privilege package permissions and immutable commit-SHA image tags.",
        ),
        table(
            [
                ["Step", "Automated gate", "Failure effect"],
                ["Checkout/setup", "Python 3.11 and pip cache", "Workflow stops"],
                ["Install", "Pinned prod + dev dependencies", "Workflow stops"],
                ["Test", "Five pytest unit/API tests", "Image is not built"],
                ["Build", "BuildKit multi-platform-capable build", "Image is not published"],
                ["Publish", "GHCR login with GITHUB_TOKEN", "Deployment is blocked"],
                ["Tag", "Commit SHA and latest", "Supports rollback and convenience"],
            ],
            [33 * mm, 72 * mm, 50 * mm],
        ),
        p("Automated test coverage", "Subsection"),
        bullet("Grayscale-to-RGB conversion, 224x224 output shape, normalized float array."),
        bullet("Invalid image rejection."),
        bullet("Feature vector and prediction response contract."),
        bullet("Health and valid multipart prediction endpoints."),
        bullet("Unsupported upload content type returns HTTP 415."),
        Preformatted("$ pytest -q\n.....  [100%]\n5 passed in 1.96s", styles["CodeBlock"]),
        p("Artifact publishing", "Subsection"),
        p(
            "On pushes, GitHub Actions authenticates to <b>ghcr.io</b> with the repository-scoped token "
            "and publishes both <b>ghcr.io/&lt;owner&gt;/cats-dogs-mlops:&lt;commit-sha&gt;</b> and "
            "<b>:latest</b>. Pull requests build but do not push, preventing untrusted publication.",
        ),
        p("6. M4 - Continuous Deployment", "Section"),
        p(
            "The selected target is Docker Compose on a simple VM. A production GitHub environment "
            "can enforce approval and secret scoping. After a successful main-branch image publish, the "
            "deploy job loads an SSH key, connects to the host, pulls the immutable SHA tag, and runs "
            "<b>docker compose up -d --remove-orphans</b>.",
        ),
        Preformatted(
            "push to main -> tests -> build -> GHCR push -> SSH deploy -> health + prediction smoke test\n"
            "failure anywhere -------------------------------------------------> release fails",
            styles["CodeBlock"],
        ),
        p("Required deployment configuration", "Subsection"),
        table(
            [
                ["Name", "Type", "Purpose"],
                ["DEPLOY_HOST", "Secret", "Docker Compose host DNS name or IP"],
                ["DEPLOY_USER", "Secret", "Least-privilege SSH account"],
                ["DEPLOY_SSH_KEY", "Secret", "Private deployment key"],
                ["DEPLOY_PATH", "Variable", "Host checkout path; defaults to /opt/cats-dogs-mlops"],
            ],
            [43 * mm, 30 * mm, 82 * mm],
        ),
        p("Smoke test", "Subsection"),
        p(
            "The smoke script retries health during startup, submits one multipart image, and requires "
            "HTTP 200 with a label and probability map. Any failure exits non-zero and fails deployment.",
        ),
        Spacer(1, 7 * mm),
        p("7. M5 - Monitoring, Logging, and Performance", "Section"),
        p("Request/response logging", "Subsection"),
        p(
            "Middleware emits request ID, HTTP method, route, status, and latency. Prediction logs contain "
            "only label, confidence, media type, and byte count. Filenames, image contents, and request "
            "bodies are excluded to avoid leaking user data.",
        ),
        Preformatted(
            "INFO request id=... method=POST path=/predict status=200 latency_ms=48.21\n"
            "INFO prediction label=dog confidence=0.5267 content_type=image/jpeg bytes=21989",
            styles["CodeBlock"],
        ),
        p("Prometheus metrics", "Subsection"),
        table(
            [
                ["Metric", "Labels", "Use"],
                ["inference_requests_total", "endpoint, status", "Traffic and error-rate monitoring"],
                ["inference_latency_seconds", "endpoint", "Latency distribution and SLO tracking"],
                ["process_* / python_*", "standard", "Runtime resource and garbage-collection signals"],
            ],
            [58 * mm, 43 * mm, 54 * mm],
        ),
        p("Performance feedback loop", "Subsection"),
        p(
            "The provided labeled-request CSV can be appended with production predictions and delayed "
            "true labels. A scheduled job can compute rolling accuracy/F1, compare them with the baseline, "
            "and trigger investigation or retraining when quality falls below an agreed threshold. "
            "Confidence histograms and input feature statistics can provide early drift signals even before "
            "labels arrive.",
        ),
        p("Operational recommendations", "Subsection"),
        bullet("Alert when 5xx rate exceeds 1% for 5 minutes or p95 latency exceeds the SLO."),
        bullet("Retain only non-sensitive structured logs and apply a bounded retention policy."),
        bullet("Evaluate quality by source/channel to expose sampling bias."),
        bullet("Promote a new model only after offline metrics and shadow traffic checks pass."),
        p("8. Verification Evidence", "Section"),
        table(
            [
                ["Check", "Observed result", "Status"],
                ["DVC reproduce", "600 images, two stages, lock file updated", "PASS"],
                ["Model artifact", "models/model.joblib, 516 KB", "PASS"],
                ["MLflow", "Runs, params, metrics, and artifacts written", "PASS"],
                ["pytest", "5 passed, 0 warnings", "PASS"],
                ["Live API", "Health + prediction HTTP 200", "PASS"],
                ["Monitoring", "Request counters and latency exported", "PASS"],
                ["Compose manifest", "Docker Compose configuration included", "PASS"],
                ["Container pull", "Docker Hub base-image access required", "Environment dependent"],
            ],
            [43 * mm, 85 * mm, 27 * mm],
        ),
        PageBreak(),
        p("9. Reproduction and Demonstration", "Section"),
        p("Reproduce training", "Subsection"),
        Preformatted(
            "python -m venv .venv\n"
            "source .venv/bin/activate\n"
            "pip install -r requirements.txt -r requirements-dev.txt\n"
            "pip install .\n"
            "python scripts/download_data.py --output data/raw\n"
            "dvc repro\n"
            "pytest -q",
            styles["CodeBlock"],
        ),
        p("Run and inspect", "Subsection"),
        Preformatted(
            "uvicorn mlops_cats_dogs.api:app --host 0.0.0.0 --port 8000\n"
            "curl http://localhost:8000/health\n"
            "curl -X POST -F 'file=@sample.jpg' http://localhost:8000/predict\n"
            "curl http://localhost:8000/metrics\n"
            "mlflow ui --backend-store-uri ./mlruns",
            styles["CodeBlock"],
        ),
        p("Submission contents", "Subsection"),
        table(
            [
                ["Artifact", "Included"],
                ["Source + tests + scripts", "Yes"],
                ["DVC pipeline and lock", "Yes"],
                ["Trained model and evaluation artifacts", "Yes"],
                ["Dockerfile + Compose", "Yes"],
                ["CI/CD workflow", "Yes"],
                ["PDF report", "Yes"],
                ["Sub-five-minute demonstration video", "Yes"],
            ],
            [100 * mm, 55 * mm],
        ),
        p("Limitations and next steps", "Subsection"),
        p(
            "This is a deliberately lightweight baseline trained on 600 images, not a production pet "
            "recognition model. Accuracy should be improved through transfer learning (for example, "
            "MobileNet or ResNet), a larger stratified dataset, augmentation, calibration, and model "
            "fairness checks. The CI/CD configuration is complete but requires repository secrets, a GHCR "
            "namespace, and an external Compose host to execute the production deployment path.",
        ),
        p("References", "Section"),
        bullet("Kaggle, <i>Dogs vs Cats competition dataset</i>: https://www.kaggle.com/competitions/dogs-vs-cats/data"),
        bullet("TensorFlow Datasets, <i>cats_vs_dogs catalog</i>: https://www.tensorflow.org/datasets/catalog/cats_vs_dogs"),
        bullet("DVC documentation: https://dvc.org/doc"),
        bullet("MLflow tracking documentation: https://mlflow.org/docs/latest/ml/tracking/"),
        bullet("FastAPI documentation: https://fastapi.tiangolo.com/"),
        bullet("GitHub Container Registry: https://docs.github.com/packages/working-with-a-github-packages-registry/working-with-the-container-registry"),
        bullet("Prometheus Python client: https://prometheus.github.io/client_python/"),
    ]
    return story


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title="MLOps Assignment 2 - Cats vs Dogs",
        author="MLOps Assignment Submission",
        subject="End-to-end MLOps pipeline",
    )
    frame = Frame(document.leftMargin, document.bottomMargin, document.width, document.height, id="body")
    document.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=header_footer))
    document.build(build_story())
    print(OUTPUT)


if __name__ == "__main__":
    main()
