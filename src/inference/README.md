# Inference Scripts

Scripts are grouped by experimental setting first and provider/model second.

```text
base/
improved/
```

This structure does not mean that each model uses a separate research method. The task schema is shared across models; provider/model folders contain call adapters and output naming for the corresponding model family.

The improved setting reads cue/context files from `outputs/method_data/` when running a full local experiment. The included `examples/demo_predictions_120/` folder provides a no-API metric check path for reviewers.
