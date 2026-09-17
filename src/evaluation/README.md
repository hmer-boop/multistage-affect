# Evaluation Scripts

Metric scripts are grouped by experimental setting:

```text
base/
cot/
improved/
```

Each setting folder contains the same metric families where applicable, plus a `run_eval_center.py` helper for full local runs. For reviewer-side smoke tests, use:

```bash
bash examples/demo_predictions/run_demo_metrics.sh
```

That demo path uses released gold labels and prediction files only, so it does not require external model APIs or unreleased raw images.
