# Data Provenance and Image Availability

This repository is prepared for manuscript review. It includes code, selected labels, sanitized metadata, cue examples, demo predictions, and image files for the ArtMajeur-sourced items.

## Image File Scope

The image items are artwork images collected from public artist or artwork websites. For the peer-review-stage repository, we use a conservative redistribution policy: image files are included for the ArtMajeur-sourced samples that have received explicit permission from the rights holder/source platform, while the remaining released items are represented through labels, sanitized metadata, titles, and source/provenance links.

The ArtMajeur permission allows the ArtMajeur-sourced samples to be made available with the dataset for non-commercial academic, research, and educational use with proper attribution to the artist and ArtMajeur.

The absence of additional image files in this repository should not be interpreted as indicating that those labeled items are invalid or that their provenance is unavailable. They remain part of the 760-item review subset and are traceable through the provenance tables.

## What Is Included

- `data/release_subset_760/labels/gold_label.json`: gold labels for the 760-item review subset.
- `data/release_subset_760/metadata/items_public.jsonl`: sanitized metadata for the same subset.
- `data/release_subset_760/images/artmajeur/`: included ArtMajeur image files.
- `data/release_subset_760/images/artmajeur_images_manifest.csv`: item-to-image manifest for the included ArtMajeur files, including repository image links in `github_image_url`.
- `data/release_subset_760/provenance/image_provenance_review.xlsx`: reviewer-facing provenance workbook sorted for lookup. The `github_image_url` column is populated for released ArtMajeur image files and left blank for items provided through source/provenance lookup only.
- `data/release_subset_760/provenance/image_provenance_review.csv`: CSV version of the same reviewer-facing provenance table.
- `data/release_subset_760/provenance/image_sources_manifest.csv`: item-level provenance worksheet, including repository image links where applicable.
- `data/release_subset_760/reports/provenance_summary.md`: current provenance-audit summary.
- `examples/cue_sample_120/`: representative stage-level cue examples.
- `examples/demo_predictions_120/`: review-demo prediction files that can be used to recompute metrics without external model calls.

## URL Fields

In `data/release_subset_760/provenance/image_sources_manifest.csv`, `source_collection_url` records an artist page, collection page, or source-site scope. This is not necessarily a link to the exact artwork.

`source_page_url` is reserved for a single-work/detail-page URL only. It is filled only when an automatic title-based lookup found a specific work-page candidate. These candidates should be visually verified before any image-file release.

Rows with blank `source_page_url` are still valid labeled data rows. The blank field means that a single-work/detail-page URL has not yet been verified.

## Review Use

Reviewers can inspect the released labels, metadata schema, cue format, included ArtMajeur images, inference scripts, and evaluation pipeline. The included demo prediction package allows metric recomputation without requiring paid API calls or access to unreleased image files.

For items that are not included as image files, reviewers can use the released URLs and titles to inspect the original websites. Researchers who need additional artwork images should obtain them from the original source websites and follow the applicable website and artist terms.
