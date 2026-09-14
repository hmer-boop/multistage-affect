# Demo Predictions 120

This folder contains a 120-item demo prediction package for GPT-4o Base, CoT, and ArtTIDE.

It is intended to let reviewers recompute metrics, inspect file schemas, and verify that Base, CoT, and ArtTIDE predictions can be evaluated through the same metric pipeline without calling external model APIs. It is not the complete prediction output used for the paper tables.

## Files

- `selected_ids.json`: item IDs in this demo set.
- `metadata/items_public.jsonl`: public metadata for the demo set.
- `labels/gold_label.json`: gold labels for the demo set.
- `cues/`: three-stage cue examples used by the ArtTIDE path.
- `predictions/base/gpt4o/`: Base predictions by stage.
- `predictions/cot/gpt4o/`: CoT predictions by stage.
- `predictions/arttide/gpt4o/`: ArtTIDE predictions by stage.
- `reports/gpt4o_demo120_metrics_vs_release760.csv`: demo metrics compared with the 760-item release subset.

