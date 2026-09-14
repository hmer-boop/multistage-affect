# Multistage Affect Reproducibility Code

This repository contains review-stage reproducibility materials for a multistage affect recognition benchmark on artwork images. It includes model inference scripts, cue-assisted inference scripts, evaluation utilities, a 760-item public label and metadata subset, a limited set of low-resolution Artmajeur image files, and a 120-item demo prediction package for quick metric checks.

The full dataset, non-Artmajeur image files, high-resolution images, private annotation workspace, and private acquisition pipeline are not included at this stage. Image access and provenance are documented in `docs/IMAGE_ACCESS_STATEMENT.md`, `docs/DATA_PROVENANCE.md`, and `data/release_subset_760/provenance/image_provenance_review.xlsx`.

## Repository Layout

```text
.
├── docs/                         # Release scope, data provenance, and image access notes
├── data/
│   └── release_subset_760/        # Public review-stage labels, metadata, provenance, and limited images
├── src/
│   ├── data_preparation/          # Metadata cleaning, caption generation, and stage-input extraction
│   ├── annotation_app/            # Five-rater annotation app with 3-second Stage 1 viewing
│   ├── cue_construction/          # Stage cue construction and cue-assisted GPT inference scripts
│   ├── inference/                 # Base and improved inference adapters by provider/model
│   └── evaluation/                # Base, CoT, and improved metric scripts
├── examples/
│   ├── schema_smoke_test/         # Small synthetic files for checking the expected schema
│   ├── cue_sample_120/            # Representative three-stage cue examples
│   └── demo_predictions_120/      # GPT-4o Base, CoT, and improved predictions for metric checks
├── reports/                       # Metric audit reports for the release subset and demo package
├── requirements.txt
└── .gitignore
```

## Environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model-running scripts require the corresponding provider credentials only when you actually run inference:

```bash
export OPENAI_API_KEY="..."
export DASHSCOPE_API_KEY="..."
export GEMINI_API_KEY="..."
export HF_TOKEN="..."
```

Do not commit local `.env` files or API credentials.

## Data Included

The review-stage subset provides labels, sanitized metadata, source/provenance tables, and a limited set of low-resolution Artmajeur images:

```text
data/release_subset_760/
├── selected_ids.json
├── labels/gold_label.json
├── metadata/items_public.jsonl
├── images_512/
│   ├── README.md
│   ├── artmajeur_images_manifest.csv
│   └── artmajeur/
├── provenance/
│   ├── image_provenance_review.xlsx
│   ├── image_provenance_review.csv
│   ├── image_sources_manifest.csv
│   └── image_source_sites.csv
└── reports/
```

Gold labels use one record per item:

```json
{
  "item_id": "sample_001",
  "stage1": ["宁静"],
  "stage2": ["宁静"],
  "stage3": ["敬畏"]
}
```

Prediction files use one record per item and may use either `stageN` or `stageN_label`:

```json
{
  "item_id": "sample_001",
  "stage1": "宁静"
}
```

Allowed affect labels:

```text
悲伤, 恐惧, 厌恶, 愤怒, 宁静, 快乐, 惊奇, 敬畏
```

## Quick Schema Smoke Test

The included smoke-test files are synthetic and are not part of the study dataset.

```bash
python src/evaluation/base/stage_label_accuracy_tsmr.py \
  --gold-base examples/schema_smoke_test/gold_label.json \
  --gold-round2 examples/schema_smoke_test/gold_label_round2_eval_legacy.json \
  --pred-stage1 examples/schema_smoke_test/pred_stage1.json \
  --pred-stage2 examples/schema_smoke_test/pred_stage2.json \
  --pred-stage3 examples/schema_smoke_test/pred_stage3.json \
  --eval-id-mode pred
```

## Demo Prediction Check

`examples/demo_predictions_120/` contains 120 review-demo items with GPT-4o Base, CoT, and improved predictions. This package lets reviewers recompute metrics and verify that the three evaluation paths share the same schema without calling external model APIs.

```bash
bash examples/demo_predictions_120/run_demo_metrics.sh
```

The expected demo-vs-paper and demo-vs-release-subset numbers are documented in:

```text
reports/demo_prediction_checks/demo_predictions_120_vs_paper_full.md
examples/demo_predictions_120/reports/gpt4o_demo120_metrics_vs_release760.csv
```

## Data Preparation

The review-stage data preparation code keeps the non-private processing path:

```bash
python src/data_preparation/clean_data_caption.py --src-dir "00 add description"
python src/data_preparation/extract_min_fields.py
python src/data_preparation/generate_image_descriptions.py --max 10
python src/data_preparation/extract_stage_inputs.py
```

Or run the wrapper:

```bash
python src/data_preparation/run_data_preparation.py --max 10
```

## Annotation App

This release includes the current five-rater annotation app. It preserves the same input and output schema while using the updated Stage 1 protocol: the raw image is shown for 3 seconds, then hidden before the rater selects the Stage 1 label.

```bash
python src/annotation_app/app_5raters_stage1_3s.py
```

Then open:

```text
http://127.0.0.1:5004/rules
```

The app reads:

```text
dataset/metadata/items_merged.jsonl
```

The app writes:

```text
results/results.jsonl
results/summary.json
results/split/shard_*.json
```

## Running Inference

Inference scripts are grouped first by experimental setting, then by provider/model. This reflects the experimental design: Base and improved settings share the same three-stage task structure, while each provider/model folder contains the corresponding call adapter.

Example Base scripts:

```bash
python src/inference/base/qwen_vl/8b_instruct/run_stage1.py
python src/inference/base/qwen_vl/8b_instruct/run_stage2.py
python src/inference/base/qwen_vl/8b_instruct/run_stage3.py
```

Example improved scripts:

```bash
python src/inference/improved/qwen_vl/8b_instruct/run_stage1.py
python src/inference/improved/qwen_vl/8b_instruct/run_stage2.py
python src/inference/improved/qwen_vl/8b_instruct/run_stage3.py
```

Cue construction and GPT improved inference scripts are under:

```text
src/cue_construction/
```

## Evaluation

Metric scripts are grouped by setting:

```text
src/evaluation/base/
src/evaluation/cot/
src/evaluation/improved/
```

For a full local baseline evaluation, use:

```bash
python src/evaluation/base/run_eval_center.py
```

For a full local improved-setting evaluation, use:

```bash
python src/evaluation/improved/run_eval_center.py
```

For CoT evaluation, use:

```bash
python src/evaluation/cot/run_eval_center.py
```

The registry inside each evaluation center maps model names to prediction paths. If your local paths differ, either adjust the registry or call individual metric scripts with explicit `--gold-*` and `--pred-*` arguments.

## Review-Stage Release Scope

This repository excludes:

- the full raw/high-resolution artwork image corpus;
- non-Artmajeur image files whose source-site permission is still unresolved;
- full metadata and full gold labels;
- complete prediction result files beyond the review-stage demo package;
- private acquisition, upload, and hosting scripts;
- generated result folders, caches, and manuscript files.

See `docs/RELEASE_SCOPE.md` for the detailed include/exclude rationale.

## License

No open-source license has been selected yet. Until a license is added, all rights are reserved by the authors. Choose a license before making a permanent public archival release.
