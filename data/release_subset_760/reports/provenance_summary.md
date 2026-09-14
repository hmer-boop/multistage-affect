# Image Provenance Summary

This report summarizes provenance matching for the 760-item review-stage subset. Raw image files are not included in this repository.

## URL Semantics

- `source_collection_url`: artist page, collection page, or source-site scope provided by the author or inferred from local folders.
- `source_page_url`: single-work/detail-page URL only. This field is left blank unless a specific work-page candidate is found.
- `exact_url_status`: matching status for the single-work/detail-page URL or source-collection inference.

## Exact Work URL and Source-Collection Status

- exact_work_url_candidate: 55
- multiple_exact_work_url_candidates: 11
- source_collection_inferred_from_local_title: 25
- no_work_url_match_on_collection_pages: 200
- blocked_or_dynamic_manual_check_needed: 277
- no_collection_url_available: 192

`source_collection_inferred_from_local_title` means no exact work-page URL was found, but the artist/collection-level source URL was inferred from local `add pic` folder names and title or translation-cache matching. These rows should still be checked manually before any image-file release.

## Source-Site Hints

- artmajeur.com: 298
- blank: 192
- daize.artron.net: 67
- artist.artron.net: 62
- huangjiannan.artron.net: 29
- hanyuchen.artron.net: 28
- mazhangcheng.artron.net: 17
- guanpuxue.artron.net: 15
- haiyang.artron.net: 11
- tiexin.artron.net: 8
- ading.artron.net: 7
- artist.artron.net; artmajeur.com: 4
- yueminjun.artron.net: 4
- baike.artron.net: 4
- artist.artron.net; guanpuxue.artron.net: 2
- chenkezhi.artron.net: 2
- changshijiang.artron.net: 2
- liushouxin.artron.net: 1
- daize.artron.net; huangjiannan.artron.net: 1
- artist.artron.net; huangjiannan.artron.net: 1
- chenkezhi.artron.net; hanyuchen.artron.net: 1
- haiyang.artron.net; mazhangcheng.artron.net: 1
- artmajeur.com; haiyang.artron.net: 1
- hanyuchen.artron.net; maokaiart.com: 1
- artmajeur.com; huangjiannan.artron.net: 1

## Files

- `../provenance/image_provenance_review.xlsx`: reviewer-facing provenance workbook sorted for lookup.
- `../provenance/image_provenance_review.csv`: CSV version of the reviewer-facing provenance workbook.
- `../provenance/image_sources_manifest.csv`: item-level provenance worksheet.
- `../provenance/image_source_sites.csv`: source sites provided by the author.
- `exact_work_url_candidates.csv`: automatically extracted single-work/detail-page candidates. Candidate rows are title-based and require visual checking before public image release.
- `provenance_local_title_inference_report.csv`: local-folder and translation-cache inference report for source-collection attribution.
- `provenance_title_match_report.csv`: local title/filename matching candidates.
- `provenance_file_hash_match_report.csv`: exact file-hash matching attempt; no exact hash matches were found, likely because release images were re-encoded.
- `provenance_manual_review_items.csv`: ambiguous, unmatched, or source-folder-unresolved items requiring manual checking before any image-file release.

## Release Guidance

Treat `exact_work_url_candidate` as a candidate until the linked page is visually checked against the local image. Treat inferred source-collection URLs as search aids, not as confirmed single-work links. Keep raw image files out of the public GitHub repository unless source URLs and reuse permissions have been verified.
