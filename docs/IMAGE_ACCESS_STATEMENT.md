# Image Access Statement

This review-stage repository includes image files for items sourced from ArtMajeur. For the remaining released items, the repository provides labels, sanitized metadata, titles, and source/provenance links rather than redistributing additional image files during peer review.

The author received explicit permission from the ArtMajeur rights holder/source platform for the ArtMajeur-sourced samples. These image files may be made available with the dataset for non-commercial academic research, academic publication, analysis, and teaching purposes with proper attribution to the artist and ArtMajeur.

For transparency, the repository provides item-level labels, sanitized metadata, and a provenance worksheet:

- `data/release_subset_760/provenance/image_provenance_review.csv`
- `data/release_subset_760/provenance/image_provenance_review.xlsx`
- `data/release_subset_760/provenance/image_sources_manifest.csv`
- `data/release_subset_760/images/artmajeur_images_manifest.csv`

The reviewer-facing provenance worksheet gives each item ID, local image filename, Chinese title, English title, one `source_url`, and, when an image file is included in this repository, a `github_image_url`.

For ArtMajeur-sourced items included as image files, the corresponding files are provided under `data/release_subset_760/images/artmajeur/`. Their `github_image_url` values point to the image files in this repository. The original `source_url` values are still kept for provenance and attribution.

For artwork items that are not included as image files, `github_image_url` is blank. Open `source_url` and search the page or site using `title_zh` and, when useful, `title_en`. When a specific work-page candidate is available, `source_url` points directly to that page. Otherwise, it points to the artist page, collection page, or broad source-site browsing page used for the supplementary image collection.

Some source websites are dynamic or protected by browser checks, so a direct command-line lookup may not resolve every item even when the artwork can still be found manually from the artist or collection page.

The included ArtMajeur images and all URL/attribution fields are provided only to support scholarly review, reproducibility checking, and non-commercial academic use. The absence of other image files reflects a conservative peer-review-stage redistribution policy, not a reduction of the released label subset. Researchers who need additional images should obtain them from the original source websites and follow the applicable website and artist terms.
