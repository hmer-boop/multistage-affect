# Release Subset 760

This folder contains the review-stage public subset metadata and gold labels for 760 selected items.

A limited set of low-resolution 512 px Artmajeur image files is included under `images_512/artmajeur/` after the author received a platform reply allowing general non-commercial academic, research, and educational use of low-resolution Artmajeur platform images with proper attribution. Image files from other sources are not included while their source-site replies remain unresolved. The `provenance/image_sources_manifest.csv` file remains the item-level source table for tracking source-site hints, source/collection URLs, exact work-page URL candidates, image URLs, license status, attribution, and notes for each item.

In the manifest, `source_collection_url` means an artist page, collection page, or source-site scope. `source_page_url` is reserved for a single-work/detail-page URL only, and remains blank unless a concrete work-page candidate has been found.

Current provenance audit status is summarized in `reports/provenance_summary.md`. The exact work-page links are title-based candidates and should be visually checked against the original local artwork before any raw image release. Blank `source_page_url` values should not be interpreted as missing data labels; they indicate that a single-work/detail-page URL has not yet been verified.

## Files

- `selected_ids.json`: item IDs included in the 760-item subset.
- `labels/gold_label.json`: three-stage gold labels for the selected items.
- `metadata/items_public.jsonl`: sanitized public metadata used by the released scripts.
- `images_512/00_IMAGE_RELEASE_NOTICE.md`: reviewer-facing note explaining why only the permitted Artmajeur image subset is included as files.
- `images_512/README.md`: note on the included low-resolution Artmajeur image files and reuse limits.
- `images_512/artmajeur/`: low-resolution 512 px image files for Artmajeur-linked items in this release subset.
- `images_512/artmajeur_images_manifest.csv`: image-file manifest mapping item IDs to the included Artmajeur image paths and source links.
- `provenance/image_provenance_review.xlsx`: reviewer-facing provenance workbook, sorted with direct work-page candidates before title-search source pages.
- `provenance/image_provenance_review.csv`: CSV version of the reviewer-facing provenance workbook.
- `provenance/image_sources_manifest.csv`: source and license audit worksheet for image provenance.
- `provenance/image_source_sites.csv`: source-site hints provided by the author.
- `reports/sample_stratum_distribution.csv`: subset label distribution.
- `reports/gold_label_instance_distribution.csv`: gold-label instance distribution.
- `reports/selection_report.json`: selection summary.
- `reports/provenance_summary.md`: provenance matching summary.
- `reports/exact_work_url_candidates.csv`: automatically extracted single-work/detail-page URL candidates.
- `reports/provenance_local_title_inference_report.csv`: source-collection attribution inferred from local folder names, title matching, and translation-cache matching.
- `reports/provenance_title_match_report.csv`: local title/filename matching candidates.
- `reports/provenance_file_hash_match_report.csv`: exact file-hash matching attempt.
- `reports/provenance_manual_review_items.csv`: items needing manual provenance review.
