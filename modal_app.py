from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

APP_NAME = "museumscat-research"
DATA_DIR = "/vol/data"
ARTIFACT_DIR = "/vol/artifacts"
MODEL_DIR = "/vol/models"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unzip", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.10.0", "torchvision==0.25.0", "transformers==4.57.1",
        "Pillow==12.1.1", "pandas>=2.2", "rapidfuzz>=3.0",
        "einops==0.8.2", "addict==2.4.0", "easydict==1.13",
        "pymupdf==1.27.2.2", "psutil>=7.0", "matplotlib>=3.10",
        "huggingface_hub>=0.35", "kaggle>=1.8",
    )
    .add_local_python_source("src")
)
volume = modal.Volume.from_name("museumscat-research", create_if_missing=True)
secrets = modal.Secret.from_name("museumscat-secrets", required=False)
app = modal.App(APP_NAME)


def _download_data() -> None:
    data = Path(DATA_DIR)
    data.mkdir(parents=True, exist_ok=True)
    if (data / "train.csv").exists() and (data / "test.csv").exists() and (data / "images").exists():
        return
    subprocess.run([
        "kaggle", "competitions", "download", "-c",
        "museumscat-specimen-collection-annotation-task", "-p", str(data), "--unzip",
    ], check=True)
    archive = data / "images.zip"
    if archive.exists() and not (data / "images").exists():
        subprocess.run(["unzip", "-q", str(archive), "-d", str(data / "images")], check=True)
    volume.commit()


def _load_model():
    import torch
    from transformers import AutoModel, AutoTokenizer
    name = "baidu/Unlimited-OCR"
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, cache_dir=MODEL_DIR)
    model = AutoModel.from_pretrained(name, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16, cache_dir=MODEL_DIR)
    return tokenizer, model.eval().cuda()


@app.function(image=image, gpu="A100-80GB", volumes={"/vol": volume}, secrets=[secrets], timeout=60 * 60 * 8)
def prepare():
    _download_data()
    import pandas as pd
    train = pd.read_csv(Path(DATA_DIR) / "train.csv")
    test = pd.read_csv(Path(DATA_DIR) / "test.csv")
    print(json.dumps({"train_rows": len(train), "test_rows": len(test), "data_dir": DATA_DIR}))


@app.function(image=image, gpu="A100-80GB", volumes={"/vol": volume}, secrets=[secrets], timeout=60 * 60 * 8)
def microcv(n_samples: int = 32):
    _download_data()
    import pandas as pd
    from src.baseline import infer_image
    from src.metrics import aurc, ned
    from src.parse_ocr import parse_ocr
    tokenizer, model = _load_model()
    train = pd.read_csv(Path(DATA_DIR) / "train.csv").head(n_samples)
    predictions, risks, confidences = [], [], []
    for row in train.itertuples(index=False):
        raw_dir = Path(ARTIFACT_DIR) / "microcv_raw" / Path(row.image_file).stem
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw = infer_image(model, tokenizer, Path(DATA_DIR) / "images" / row.image_file, raw_dir)
        parsed = parse_ocr(raw)
        risk = 0.5 * ned(parsed.date, row.verbatimDate, date=True) + 0.5 * ned(parsed.locality, row.verbatimLocality)
        confidence = min(parsed.date_confidence, parsed.locality_confidence)
        predictions.append({"image_file": row.image_file, "pred_date": parsed.date, "truth_date": row.verbatimDate, "pred_locality": parsed.locality, "truth_locality": row.verbatimLocality, "risk": risk, "confidence": confidence})
        risks.append(risk)
        confidences.append(confidence)
    result = {"n": len(predictions), "mean_risk": sum(risks) / max(len(risks), 1), "aurc": aurc(risks, confidences), "predictions": predictions}
    out = Path(ARTIFACT_DIR) / f"microcv_{n_samples}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    volume.commit()
    print(json.dumps({k: result[k] for k in ("n", "mean_risk", "aurc")}))


@app.function(image=image, gpu="A100-80GB", volumes={"/vol": volume}, secrets=[secrets], timeout=60 * 60 * 12)
def baseline(limit: int | None = None):
    _download_data()
    from src.baseline import run_baseline
    tokenizer, model = _load_model()
    out = run_baseline(model, tokenizer, DATA_DIR, ARTIFACT_DIR, limit=limit)
    volume.commit()
    print(out)


@app.function(image=image, volumes={"/vol": volume}, timeout=60 * 10)
def download_artifact(artifact: str = "submission_unlimited_ocr.csv"):
    path = Path(ARTIFACT_DIR) / artifact
    if not path.exists():
        raise FileNotFoundError(path)
    print(path.read_text())


@app.function(image=image, volumes={"/vol": volume}, secrets=[secrets], timeout=60 * 10)
def submit_lb(artifact: str = "submission_unlimited_ocr.csv", allow_lb_submit: bool = False):
    if not allow_lb_submit:
        raise RuntimeError("LB submission is approval-gated. Re-run only after explicit approval with --allow-lb-submit.")
    _download_data()
    path = Path(ARTIFACT_DIR) / artifact
    subprocess.run([
        "kaggle", "competitions", "submit", "-c",
        "museumscat-specimen-collection-annotation-task", "-f", str(path),
        "-m", "Unlimited-OCR baseline",
    ], check=True)


if __name__ == "__main__":
    print("Use: modal run modal_app.py::prepare | ::microcv | ::baseline | ::download_artifact | ::submit_lb")
