# Multistage Affect Reproducibility Code

<p align="center">
  Reproducibility materials for multistage affect recognition on artwork images
</p>

<p align="center">
  <img src="assets/readme/artwork_samples.png" alt="Representative artwork samples" width="100%">
</p>

<p align="center"><em>Representative artwork samples. Copyright remains with the respective artists; source and attribution information is provided in the image manifest.</em></p>

---

## ✨ Overview

This repository contains code and supporting materials for a multistage affect recognition benchmark on artwork images. It provides data-processing and annotation utilities, model inference adapters, cue-assisted inference code, evaluation scripts, human gold labels for a representative subset, sanitized metadata, provenance records, and a compact prediction package for metric verification.

The repository currently provides a representative subset rather than the complete benchmark. Release scope, future availability, image access, and provenance are documented in [Review-Stage Release Scope](docs/RELEASE_SCOPE.md), [Image Access Statement](docs/IMAGE_ACCESS_STATEMENT.md), [Data Provenance](docs/DATA_PROVENANCE.md), and the [provenance workbook](data/release_subset/provenance/image_provenance_review.xlsx).

## 🚀 Quick Start

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The included demonstration metrics can be recomputed without model API credentials or access to the unreleased image corpus:

```bash
bash examples/demo_predictions/run_demo_metrics.sh
```

This command uses the released gold labels and saved Base, CoT, and ArtTIDE predictions. Recomputed outputs are written to `examples/demo_predictions/reports/recomputed/`, which is ignored by git.

## 🔍 Reproducibility Coverage

- Inspect the released human gold labels, sanitized metadata, and three-stage label schema.
- Examine representative cue examples used by the cue-assisted inference path.
- Recompute Base, CoT, and ArtTIDE metrics from the included demonstration predictions without paid API calls.
- Inspect the inference adapters and the shared evaluation implementation.
- Review included artwork files and trace other released items through their titles and source records.
- Run the annotation interface and inspect its input and output schema.

## 📁 Repository Structure

```text
.
├── assets/                        # README artwork strip and presentation assets
├── docs/                          # Release scope, provenance, and image access notes
├── data/
│   └── release_subset/        # Review-stage labels, metadata, provenance, and images
├── src/
│   ├── data_preparation/          # Metadata cleaning and stage-input preparation
│   ├── annotation_app/            # Five-rater annotation interface
│   ├── cue_construction/          # Three-stage cue construction
│   ├── inference/                 # Base and cue-assisted model adapters
│   └── evaluation/                # Base, CoT, and ArtTIDE metric scripts
├── examples/
│   ├── schema_smoke_test/         # Synthetic schema-check files
│   ├── cue_sample/            # Representative three-stage cue examples
│   └── demo_predictions/      # Saved predictions for metric verification
├── reports/                       # Metric audits and comparison reports
└── requirements.txt
```

## 🧾 Data and Image Access

The repository provides a representative review subset with human gold labels, sanitized metadata, provenance tables, and artwork files where direct redistribution is supported by the documented permission. Other released items remain traceable through title and source information.

ArtMajeur has provided written confirmation permitting the inclusion of ArtMajeur-sourced images in this non-commercial academic release with proper attribution. Copyright in each artwork remains with the respective artist. The image files are not covered by a general software license and must not be interpreted as granting unrestricted commercial or downstream redistribution rights.

Detailed documentation:

- [Review-stage release scope](docs/RELEASE_SCOPE.md)
- [Data provenance and image availability](docs/DATA_PROVENANCE.md)
- [Image access statement](docs/IMAGE_ACCESS_STATEMENT.md)
- [Released subset guide](data/release_subset/README.md)
- [Image manifest](data/release_subset/images/artmajeur_images_manifest.csv)

## 🧪 Reproducibility Paths

### Schema Smoke Test

The smoke-test files are synthetic and are not part of the study dataset.

```bash
python src/evaluation/base/stage_label_accuracy_tsmr.py \
  --gold-base examples/schema_smoke_test/gold_label.json \
  --gold-round2 examples/schema_smoke_test/gold_label_round2_eval_legacy.json \
  --pred-stage1 examples/schema_smoke_test/pred_stage1.json \
  --pred-stage2 examples/schema_smoke_test/pred_stage2.json \
  --pred-stage3 examples/schema_smoke_test/pred_stage3.json \
  --eval-id-mode pred
```

### Demonstration Metric Check

The included prediction package supports inspection of the common prediction schema and recomputation of the reported metric families without calling external models.

```bash
bash examples/demo_predictions/run_demo_metrics.sh
```

Reference reports are available at:

- [Demo-to-paper metric comparison](reports/demo_prediction_checks/demo_predictions_vs_paper_full.md)
- [Demo-to-review-subset metric comparison](examples/demo_predictions/reports/gpt4o_demo_metrics_vs_release_subset.csv)

## 🛠️ Data Preparation

The repository retains the non-private processing path used to clean metadata and prepare stage inputs:

```bash
python src/data_preparation/run_data_preparation.py --max 10
```

Individual processing utilities are located in [`src/data_preparation/`](src/data_preparation/).

## 🗂️ Annotation Protocol

The included annotation interface implements the five-rater protocol. During Stage 1, the artwork is displayed for three seconds and then hidden before the rater selects a label. The input and output schema is retained across all stages.

```bash
python src/annotation_app/app_5raters_stage1_3s.py
```

Then open `http://127.0.0.1:5004/rules` in a browser. The application expects input under `dataset/metadata/` and writes annotation results under `results/`.

## 🤖 Running Model Inference

Inference requires credentials only when calling external model providers. Set the credentials for the providers you intend to use:

```bash
export OPENAI_API_KEY="..."
export DASHSCOPE_API_KEY="..."
export GEMINI_API_KEY="..."
export HF_TOKEN="..."
```

Do not commit local `.env` files or credentials. Base and cue-assisted scripts share the same three-stage task schema; provider and model folders contain the corresponding call adapters. See [`src/inference/README.md`](src/inference/README.md) for the folder conventions and entry points.

## 📐 Evaluation

Evaluation scripts are organized by experimental setting:

```text
src/evaluation/base/
src/evaluation/cot/
src/evaluation/improved/
```

Each folder contains the applicable metric implementations and a `run_eval_center.py` helper for full local experiments. The included demonstration command is the recommended quick verification entry point. See [`src/evaluation/README.md`](src/evaluation/README.md) for details.

## 📖 Citation

Citation information will be added after publication. This repository will be updated with the article citation, persistent identifier, and full Hugging Face dataset link when they become available.

## ⚖️ Usage and Copyright

No open-source license is granted for this repository. Unless otherwise stated, the code, annotations, metadata, and documentation are provided for scholarly use and reproducibility inspection, with all rights reserved by their respective copyright holders.

Copyright in the artwork images remains with the respective artists. Their inclusion does not create a blanket license for commercial use or unrestricted redistribution. Consult the image manifest, source records, and [Image Access Statement](docs/IMAGE_ACCESS_STATEMENT.md) for attribution and use information.

## 📬 Questions and Rights Concerns

For technical questions or rights-related concerns, please open a GitHub issue. Author, publication, citation, and archival dataset information will be added after acceptance.
