# Data Provenance and Image Availability

This repository is prepared for manuscript review. It includes code, selected labels, sanitized metadata, cue examples, demo predictions, and a limited set of low-resolution Artmajeur image files.

## Image File Scope

The image items are artwork images collected from public artist or artwork websites. Because source permissions differ by platform and artist, this review-stage release includes only the low-resolution 512 px files for Artmajeur-linked items whose platform permission has been clarified by the author.

The Artmajeur platform reply allows general non-commercial academic, research, and educational use of low-resolution images available on Artmajeur with proper attribution to the artist and Artmajeur. Copyright remains with each individual artist, and broader use or higher-resolution reuse should be requested from the artist directly.

Image files from other source sites are not included while their source-site replies remain unresolved. Those items remain link-only in the provenance tables.

## What Is Included

- `data/release_subset_760/labels/gold_label.json`: gold labels for the 760-item review subset.
- `data/release_subset_760/metadata/items_public.jsonl`: sanitized metadata for the same subset.
- `data/release_subset_760/images_512/artmajeur/`: included low-resolution Artmajeur image files.
- `data/release_subset_760/images_512/artmajeur_images_manifest.csv`: item-to-image manifest for the included Artmajeur files.
- `data/release_subset_760/provenance/image_provenance_review.xlsx`: reviewer-facing provenance workbook sorted for lookup.
- `data/release_subset_760/provenance/image_provenance_review.csv`: CSV version of the same reviewer-facing provenance table.
- `data/release_subset_760/provenance/image_sources_manifest.csv`: item-level provenance worksheet.
- `data/release_subset_760/reports/provenance_summary.md`: current provenance-audit summary.
- `examples/cue_sample_120/`: representative stage-level cue examples.
- `examples/demo_predictions_120/`: review-demo prediction files that can be used to recompute metrics without external model calls.

## URL Fields

In `data/release_subset_760/provenance/image_sources_manifest.csv`, `source_collection_url` records an artist page, collection page, or source-site scope. This is not necessarily a link to the exact artwork.

`source_page_url` is reserved for a single-work/detail-page URL only. It is filled only when an automatic title-based lookup found a specific work-page candidate. These candidates should be visually verified before any image-file release.

Rows with blank `source_page_url` are still valid labeled data rows. The blank field means that a single-work/detail-page URL has not yet been verified.

## Review Use

Reviewers can inspect the released labels, metadata schema, cue format, included low-resolution Artmajeur images, inference scripts, and evaluation pipeline. The included demo prediction package allows metric recomputation without requiring paid API calls or access to unreleased image files.

For source sites that are not included as image files, reviewers can use the released URLs and titles to inspect the original websites. Researchers who need additional artwork images should obtain them from the original source websites and follow the applicable website and artist terms.
