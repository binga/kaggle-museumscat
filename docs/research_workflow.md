# MuseumSCAT research workflow

## Objective

Optimize normalized edit distance while ranking predictions by confidence so that easy, accurate transcriptions are covered first. The task has only 200 labeled training examples, 3,300 test images, difficult handwriting, multiple labels per specimen, and exact-transcription requirements.

## Phase 0 — data and baseline audit

1. Download and checksum Kaggle data into a Modal Volume.
2. Verify schemas and image coverage.
3. Run Unlimited-OCR on a fixed 32-image smoke set.
4. Save raw OCR, parsed fields, latency, errors, and confidence components.
5. Generate—but do not submit—the baseline CSV.

## Phase 1 — MicroCV

MicroCV uses labeled training data and never touches the Kaggle leaderboard.

- Fixed, reviewable validation manifest.
- Date and locality normalized edit distance separately.
- Joint row quality and approximate AURC.
- Missing values, pipe-separated cards, and parse failures tracked separately.
- Start with 32 examples, then 64 and 128 only after the smoke run is healthy.

## Phase 2 — candidate experiments on Modal

Every candidate is a versioned config and produces a JSON run record:

1. Prompt variants: strict JSON, field-specific, and card-aware prompts.
2. Image variants: original, crop/resize, contrast normalization, and multi-view crops.
3. Decoding variants: deterministic decoding, bounded retries, and self-consistency only if MicroCV supports it.
4. Parser variants: robust JSON extraction, delimiter repair, date normalization, and locality cleanup.
5. Confidence variants: output agreement, parse validity, field-level uncertainty, and calibration on held-out labels.
6. Optional supervised adapters only after the zero-shot baseline is understood.

Modal is the only execution backend. Each run records the image manifest, git commit, model revision, Modal image digest, config, seed, timings, raw outputs, parsed outputs, metrics, and artifact paths.

## Phase 3 — controlled leaderboard probes

Leaderboard submissions are reserved for questions local validation cannot answer. Each probe changes one hypothesis at a time, receives a submission ID, and is recorded with the Kaggle score. Never spend a submission merely to check that a file uploads.

## Approval gates

- **Gate A:** repository, data manifest, and baseline design.
- **Gate B:** first MicroCV run.
- **Gate C:** baseline test predictions.
- **Gate D:** first Kaggle leaderboard submission.

The current implementation stops before Gate D.
