# Review-Stage Release Scope

This folder was prepared as a conservative GitHub release for manuscript submission.

## Included

- Baseline inference scripts for GPT-4o, Gemini, Qwen3-VL, InternVL, and LLaVA-OneVision.
- Improved inference scripts where the manuscript discusses the ArtTIDE-style cue / context-assisted procedure.
- The current five-rater annotation app with 3-second Stage 1 viewing and unchanged result schema.
- Baseline, improved, and CoT evaluation scripts.
- Synthetic sample files documenting expected label and prediction schemas.
- A 760-item review-stage public subset with sanitized metadata and gold labels.
- Image files for ArtMajeur-sourced items in the 760-item subset, accompanied by an image manifest and attribution/provenance tables.
- A 120-item cue sample covering all three stages.
- A 120-item demo prediction package for GPT-4o Base, CoT, and ArtTIDE metric checks.
- Metric audit reports comparing the paper table, 760-item subset, and demo prediction package.
- An image source/license audit manifest for tracking provenance before any image-file release.
- A reviewer-facing image provenance workbook and CSV, sorted with direct work-page candidates first and title-search source pages afterward.
- A data provenance note explaining which image files are included and how the remaining items can be traced through metadata, titles, and source links.
- An image access statement explaining how reviewers can locate artwork items from the released source pages and titles.
- A `.gitignore` designed to keep raw data, generated predictions, checkpoints, caches, and credentials out of Git.

## Excluded

- Full raw/high-resolution artwork images.
- Additional artwork image files beyond the currently released ArtMajeur-sourced subset.
- Full metadata, full gold labels, and complete prediction result files beyond the review-stage public subset.
- Earlier local annotation variants, annotation working files, and manually curated labeling artifacts.
- Data scraping, acquisition, upload, and hosting scripts.
- Manuscript drafts, cover letters, language audits, temporary files, IDE metadata, virtual environments, and cache files.

## Rationale

The goal is to let reviewers inspect and reproduce the experimental logic without exposing the complete private dataset during review. The image-file subset reflects a conservative redistribution choice for the review stage; the remaining released items are still traceable through metadata, titles, and source links. The full dataset and additional artifacts can be released after acceptance, subject to copyright, consent, and journal policy constraints.

## Before Public Upload

Run these checks from the release folder:

```bash
find . -type f -size +50M
rg -n '/Users/|[A-Z]:\\|OPENAI_API_KEY=.*[A-Za-z0-9]|DASHSCOPE_API_KEY=.*[A-Za-z0-9]|GEMINI_API_KEY=.*[A-Za-z0-9]|HF_TOKEN=.*[A-Za-z0-9]'
rg -n "images_raw|dataset_hf_upload|gold_label.json|summary.json"
```

Mentions of expected data directories and environment variable names are fine; committed actual files or real secret values are not.
