# ArtTIDE Reproducibility Code

This repository contains the reproducibility code for the ArtTIDE experiments. It is prepared for manuscript review and includes model inference scripts, improved prompting / cue-based inference scripts, and evaluation utilities.

The full dataset, raw artwork images, annotation workspace, and private data acquisition pipeline are not included at this stage. A small synthetic sample is provided only to document the expected file formats and to support quick smoke tests of the evaluation scripts.

## Repository Contents

```text
.
├── 8_run_label1.py
├── 8_run_label2.py
├── 8_run_label3.py
├── 1_clean data caption.py       # Initial item metadata and title translation
├── 3_extract_min_fields.py       # Minimal stage-1/2 input extraction
├── 4_run describe.py             # Objective image caption generation
├── 5_all.py                      # Data-preparation runner
├── 6.1.2_app_5raters_stage1_3s.py # Five-rater annotation app with 3-second Stage 1 viewing
├── 7_extract data.py             # Stage input extraction from merged metadata
├── 000test_qwen/                 # Qwen3-VL baseline and improved inference scripts
├── 000test_InternVL/             # InternVL baseline and improved inference scripts
├── 000test_llava/                # LLaVA-OneVision baseline and improved inference scripts
├── 000test_gemini/               # Gemini baseline and improved inference scripts
├── 01improvement method/         # ArtTIDE cue construction and improved GPT inference
├── 9_evaluation/                 # Baseline evaluation scripts
├── 9.2_evaluation_improve/       # Improved-method evaluation scripts
├── 9.3_evaluation_ablation/      # Ablation evaluation scripts
├── 9.4_evaluation_cot/           # Chain-of-thought evaluation scripts
└── examples/sample_data/         # Synthetic examples for schema and smoke tests
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

## Data Layout

The private full dataset is intentionally omitted. For a full run, place your own data using the same schema:

```text
dataset/
├── images_raw/
│   └── <item_id>.jpg
└── metadata/
    ├── items_stage1_2.json
    ├── items_stage1_2_round2.json
    └── items_min.jsonl

results/
├── gold_label.json
└── gold_label_round2_eval_legacy.json
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

## Quick Evaluation Smoke Test

The included sample data are synthetic and are not part of the study dataset.

```bash
python 9_evaluation/9.1_stage_label_accuracy_tsmr_cli.py \
  --gold-base examples/sample_data/gold_label.json \
  --gold-round2 examples/sample_data/gold_label_round2_eval_legacy.json \
  --pred-stage1 examples/sample_data/pred_stage1.json \
  --pred-stage2 examples/sample_data/pred_stage2.json \
  --pred-stage3 examples/sample_data/pred_stage3.json \
  --eval-id-mode pred
```

For the full baseline evaluation, use:

```bash
python 9_evaluation/run_eval_center.py
```

For the improved-method evaluation, use:

```bash
python 9.2_evaluation_improve/run_eval_center_improve.py
```

The registry inside each evaluation center maps model names to prediction paths. If your local paths differ, either adjust the registry or call individual metric scripts with explicit `--gold-*` and `--pred-*` arguments.

## Data Preparation

The review-stage data preparation code keeps only the non-annotation path:

```bash
python "1_clean data caption.py" --src-dir "00 add description"
python "3_extract_min_fields.py"
python "4_run describe.py" --max 10
python "7_extract data.py"
```

Or run the wrapper:

```bash
python "5_all.py" --max 10
```

## Annotation App

This release includes the current five-rater annotation app. It preserves the same input and output schema while using the updated Stage 1 protocol: the raw image is shown for 3 seconds, then hidden before the rater selects the Stage 1 label.

```bash
python "6.1.2_app_5raters_stage1_3s.py"
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

Earlier local annotation variants, annotation work logs, and manual gold-labeling scripts are intentionally excluded.

## Running Inference

Baseline scripts are grouped by model and stage. For example:

```bash
python 000test_qwen/8b-instruct/1_baseline/8_qwen8b_run_label1.py
python 000test_qwen/8b-instruct/1_baseline/8_qwen8b_run_label2.py
python 000test_qwen/8b-instruct/1_baseline/8_qwen8b_run_label3.py
```

Improved scripts are under each model's `2_improve/` directory. The GPT-based ArtTIDE cue and improved inference pipeline is under `01improvement method/`.

## Review-Stage Release Scope

This repository is a review-stage code release. It excludes:

- raw artwork images;
- full metadata and full gold labels;
- earlier local annotation variants and annotation work logs;
- web scraping / data acquisition scripts;
- private upload scripts;
- generated result folders, caches, and manuscript files.

See `RELEASE_SCOPE.md` for the detailed include/exclude rationale.

## License

No open-source license has been selected yet. Until a license is added, all rights are reserved by the authors. Choose a license before making a permanent public archival release.
