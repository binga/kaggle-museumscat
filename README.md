# MuseumSCAT — Unlimited-OCR baseline

Research repository for the [MuseumSCAT: Specimen Collection Annotation Task @ CVNH ECCV26](https://www.kaggle.com/competitions/museumscat-specimen-collection-annotation-task).

The first baseline uses [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR) to transcribe each specimen-label image into the required `verbatimDate`, `verbatimLocality`, and confidence fields.

## Guardrails

- All experiments run on Modal. Local code only launches jobs and reads artifacts.
- No Kaggle submission is made by default.
- `submit-lb` requires an explicit `--allow-lb-submit` flag and a separate approval checkpoint.
- Kaggle/Hugging Face credentials are read from Modal Secrets; never commit credentials or data.

## Competition data

The competition provides `train.csv` (200 labeled examples), `test.csv` (3,300 image filenames), and `images/` (3,500 JPEG images). The submission columns are:

```text
image_file,verbatimDate,verbatimDate_confidence,verbatimLocality,verbatimLocality_confidence
```

## Setup

Create a Modal Secret named `museumscat-secrets` containing `KAGGLE_USERNAME`, `KAGGLE_KEY`, and optionally `HF_TOKEN`. Then install the project locally:

```bash
uv sync
```

The first Modal run downloads the competition data and caches the model on a persistent Modal Volume.

## Workflow

```bash
uv run modal run modal_app.py::prepare
uv run modal run modal_app.py::microcv --n-samples 32
uv run modal run modal_app.py::baseline
uv run modal run modal_app.py::download_artifact --artifact submission_unlimited_ocr.csv
```

The `submit-lb` function is intentionally protected and must not be run before approval:

```bash
uv run modal run modal_app.py::submit_lb --artifact submission_unlimited_ocr.csv --allow-lb-submit
```

See [`docs/research_workflow.md`](docs/research_workflow.md) for the MicroCV → Modal experiment → LB-probe protocol.
