# Image Access Statement

This review-stage repository includes a limited set of low-resolution 512 px artwork images from Artmajeur-linked items. Other artwork image files are not redistributed in this repository because their source-site replies remain unresolved.

The author received a written platform reply from the Artmajeur team stating that low-resolution images available on Artmajeur may generally be used for non-commercial academic research, academic publication, analysis, and teaching purposes with proper attribution to the artist and Artmajeur. The reply also notes that copyright for each artwork remains with the individual artist, and that higher-resolution or broader permissions should be requested from the relevant artist directly.

For transparency, the repository provides item-level labels, sanitized metadata, and a provenance worksheet:

- `data/release_subset_760/provenance/image_provenance_review.csv`
- `data/release_subset_760/provenance/image_provenance_review.xlsx`
- `data/release_subset_760/provenance/image_sources_manifest.csv`
- `data/release_subset_760/images_512/artmajeur_images_manifest.csv`

The reviewer-facing provenance worksheet gives each item ID, local image filename, Chinese title, English title, and one `source_url`.

For Artmajeur-linked items included as image files, the corresponding 512 px files are provided under `data/release_subset_760/images_512/artmajeur/`. The table links are still kept for provenance and attribution.

For other artwork items, open `source_url` and search the page or site using `title_zh` and, when useful, `title_en`. When a specific work-page candidate is available, `source_url` points directly to that page. Otherwise, it points to the artist page, collection page, or broad source-site browsing page used for the supplementary image collection.

Some source websites are dynamic or protected by browser checks, so a direct command-line lookup may not resolve every item even when the artwork can still be found manually from the artist or collection page.

The included Artmajeur images and all URL/attribution fields are provided only to support scholarly review, reproducibility checking, and non-commercial academic use. They should not be treated as a blanket license for commercial use, high-resolution reuse, or redistribution outside the stated academic context. Researchers who need additional images should obtain them from the original source websites and follow the applicable website and artist terms.
