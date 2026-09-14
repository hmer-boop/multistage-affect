# Release Subset 760

This folder contains the review-stage public subset metadata and gold labels for 760 selected items.

Image files for the ArtMajeur-sourced items are included under `images/artmajeur/` because these samples have received explicit permission from the rights holder/source platform and can be made available with the dataset for non-commercial academic use with proper attribution. For the peer-review-stage repository, other artwork images are provided as source metadata and retrieval links rather than redistributed image files. This is a conservative release policy and should not be interpreted as indicating that the corresponding labeled items are invalid or unusable. The `provenance/image_sources_manifest.csv` file remains the item-level source table for tracking source-site hints, source/collection URLs, exact work-page URL candidates, image URLs, attribution, and notes for each item.

In the manifest, `source_collection_url` means an artist page, collection page, or source-site scope. `source_page_url` is reserved for a single-work/detail-page URL only, and remains blank unless a concrete work-page candidate has been found.

Current provenance audit status is summarized in `reports/provenance_summary.md`. The exact work-page links are title-based candidates and should be visually checked against the original local artwork before any raw image release. Blank `source_page_url` values should not be interpreted as missing data labels; they indicate that a single-work/detail-page URL has not yet been verified.

## Files

- `selected_ids.json`: item IDs included in the 760-item subset.
- `labels/gold_label.json`: three-stage gold labels for the selected items.
- `metadata/items_public.jsonl`: sanitized public metadata used by the released scripts.
- `images/00_IMAGE_RELEASE_NOTICE.md`: reviewer-facing note explaining the image-file release scope.
- `images/README.md`: note on the included ArtMajeur image files and reuse limits.
- `images/artmajeur/`: image files for ArtMajeur-sourced items in this release subset.
- `images/artmajeur_images_manifest.csv`: image-file manifest mapping item IDs to the included ArtMajeur image paths and source links.
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
