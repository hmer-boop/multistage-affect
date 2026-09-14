# GitHub Release Metric Audit

This audit compares the paper-level full-data metrics with the selected 760-item release subset. Values are formatted as `paper full / release subset / delta`, where delta is `release subset - paper full`.

## Release Subset

- Selected items: 760
- Image folder size: about 362M
- Positive / negative: 426 / 334 = 56.05% / 43.95%
- Public gold labels: `labels/gold_label_sample.json`
- Public metadata: `metadata/items_sample_public.jsonl`
- Review examples: `examples/cue_sample_120` and `examples/demo_predictions_120`

## Full vs 760-Item Release Subset

| Model | Setting | S1L | S2L | S3L | TSMR | CSGR | CRR | ML-MF1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-4o | Base | 0.3870 / 0.3789 / -0.0081 | 0.4040 / 0.4026 / -0.0014 | 0.4750 / 0.4868 / +0.0118 | 0.4220 / 0.4228 / +0.0008 | 0.4090 / 0.4207 / +0.0117 | 0.4270 / 0.4150 / -0.0120 | 0.3490 / 0.3432 / -0.0058 |
| GPT-4o | CoT | 0.4200 / 0.4211 / +0.0011 | 0.4280 / 0.4329 / +0.0049 | 0.4960 / 0.4974 / +0.0014 | 0.4480 / 0.4504 / +0.0024 | 0.3280 / 0.3295 / +0.0015 | 0.2800 / 0.2827 / +0.0027 | 0.3720 / 0.3787 / +0.0067 |
| GPT-4o | ArtTIDE | 0.4730 / 0.4776 / +0.0046 | 0.5090 / 0.5000 / -0.0090 | 0.5660 / 0.5632 / -0.0028 | 0.5160 / 0.5136 / -0.0024 | 0.4370 / 0.4395 / +0.0025 | 0.3100 / 0.3132 / +0.0032 | 0.4260 / 0.4163 / -0.0097 |
| Gemini-2.5-Pro | Base | 0.3960 / 0.4013 / +0.0053 | 0.4400 / 0.4592 / +0.0192 | 0.5120 / 0.5224 / +0.0104 | 0.4490 / 0.4610 / +0.0120 | 0.4080 / 0.4063 / -0.0017 | 0.3550 / 0.3410 / -0.0140 | 0.3710 / 0.3736 / +0.0026 |
| Gemini-2.5-Pro | CoT | 0.4440 / 0.4434 / -0.0006 | 0.4890 / 0.4882 / -0.0008 | 0.5600 / 0.5627 / +0.0027 | 0.4990 / 0.5046 / +0.0056 | 0.4860 / 0.4885 / +0.0025 | 0.3620 / 0.3600 / -0.0020 | 0.4070 / 0.4167 / +0.0097 |
| Gemini-2.5-Pro | ArtTIDE | 0.4670 / 0.4860 / +0.0190 | 0.5480 / 0.5434 / -0.0046 | 0.6040 / 0.6045 / +0.0005 | 0.5400 / 0.5458 / +0.0058 | 0.5180 / 0.5029 / -0.0151 | 0.3250 / 0.3105 / -0.0145 | 0.4460 / 0.4418 / -0.0042 |
| InternVL-8B | Base | 0.3580 / 0.3430 / -0.0150 | 0.3920 / 0.3671 / -0.0249 | 0.4480 / 0.4237 / -0.0243 | 0.3990 / 0.3782 / -0.0208 | 0.3790 / 0.3451 / -0.0339 | 0.4460 / 0.4409 / -0.0051 | 0.3300 / 0.3065 / -0.0235 |
| InternVL-8B | CoT | 0.3880 / 0.3711 / -0.0169 | 0.4300 / 0.4053 / -0.0247 | 0.4740 / 0.4711 / -0.0029 | 0.4300 / 0.4158 / -0.0142 | 0.4500 / 0.4624 / +0.0124 | 0.4940 / 0.5162 / +0.0222 | 0.3550 / 0.3370 / -0.0180 |
| InternVL-8B | ArtTIDE | 0.4350 / 0.4526 / +0.0176 | 0.4740 / 0.4671 / -0.0069 | 0.5450 / 0.5461 / +0.0011 | 0.4850 / 0.4886 / +0.0036 | 0.4580 / 0.4568 / -0.0012 | 0.3590 / 0.3521 / -0.0069 | 0.4000 / 0.3960 / -0.0040 |
| LLaVA-4B | Base | 0.3210 / 0.3153 / -0.0057 | 0.3780 / 0.3803 / +0.0023 | 0.4390 / 0.4474 / +0.0084 | 0.3790 / 0.3808 / +0.0018 | 0.3790 / 0.3864 / +0.0074 | 0.4640 / 0.4533 / -0.0107 | 0.3160 / 0.3125 / -0.0035 |
| LLaVA-4B | CoT | 0.3610 / 0.3553 / -0.0057 | 0.3980 / 0.4079 / +0.0099 | 0.4570 / 0.4776 / +0.0206 | 0.4050 / 0.4136 / +0.0086 | 0.3620 / 0.3822 / +0.0202 | 0.3980 / 0.3839 / -0.0141 | 0.3350 / 0.3378 / +0.0028 |
| LLaVA-4B | ArtTIDE | 0.4150 / 0.4145 / -0.0005 | 0.4530 / 0.4519 / -0.0011 | 0.5430 / 0.5211 / -0.0219 | 0.4700 / 0.4620 / -0.0080 | 0.5080 / 0.4784 / -0.0296 | 0.4150 / 0.4286 / +0.0136 | 0.3880 / 0.3746 / -0.0134 |
| Qwen-VL-8B | Base | 0.3250 / 0.3303 / +0.0053 | 0.3750 / 0.3671 / -0.0079 | 0.4570 / 0.4645 / +0.0075 | 0.3860 / 0.3873 / +0.0013 | 0.3660 / 0.3617 / -0.0043 | 0.3910 / 0.3584 / -0.0326 | 0.3190 / 0.3140 / -0.0050 |
| Qwen-VL-8B | CoT | 0.3650 / 0.3658 / +0.0008 | 0.4130 / 0.4066 / -0.0064 | 0.4820 / 0.4947 / +0.0127 | 0.4200 / 0.4224 / +0.0024 | 0.4240 / 0.4501 / +0.0261 | 0.4350 / 0.4401 / +0.0051 | 0.3470 / 0.3423 / -0.0047 |
| Qwen-VL-8B | ArtTIDE | 0.4170 / 0.4395 / +0.0225 | 0.4930 / 0.4855 / -0.0075 | 0.5360 / 0.5441 / +0.0081 | 0.4820 / 0.4897 / +0.0077 | 0.4700 / 0.4590 / -0.0110 | 0.3860 / 0.3659 / -0.0201 | 0.3980 / 0.3969 / -0.0011 |

## Demo Predictions 120 vs 760-Item Release Subset

Demo predictions include GPT-4o Base, CoT, and ArtTIDE predictions for 120 items. Values are `release760 / demo120 / delta`, where delta is `demo120 - release760`.

| Model | Setting | S1L | S2L | S3L | TSMR | CSGR | CRR | ML-MF1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-4o | Base | 0.3789 / 0.3750 / -0.0039 | 0.4026 / 0.3917 / -0.0109 | 0.4868 / 0.4750 / -0.0118 | 0.4228 / 0.4139 / -0.0089 | 0.4207 / 0.4110 / -0.0097 | 0.4150 / 0.4255 / +0.0105 | 0.3432 / 0.3394 / -0.0038 |
| GPT-4o | CoT | 0.4211 / 0.4000 / -0.0211 | 0.4329 / 0.4333 / +0.0004 | 0.4974 / 0.5167 / +0.0193 | 0.4504 / 0.4500 / -0.0004 | 0.3295 / 0.3529 / +0.0234 | 0.2827 / 0.2692 / -0.0135 | 0.3787 / 0.3731 / -0.0056 |
| GPT-4o | ArtTIDE | 0.4776 / 0.4833 / +0.0057 | 0.5000 / 0.4833 / -0.0167 | 0.5632 / 0.5500 / -0.0132 | 0.5136 / 0.5056 / -0.0080 | 0.4395 / 0.4194 / -0.0201 | 0.3132 / 0.3103 / -0.0029 | 0.4163 / 0.4108 / -0.0055 |

## Reviewer-Visible Checks

- Inspect the public 760-item image subset, public metadata, and gold labels.
- Recompute release-subset metrics from provided gold labels and demo prediction files without calling external model APIs.
- Verify that Base, CoT, and ArtTIDE use the same input schema and evaluation scripts.
- Inspect 120 representative stage cue examples across stages 1, 2, and 3.
- Run new inference with their own API keys on the public subset or the 120-item demo subset, then compare against the included metric scripts.

## Limits

- The demo predictions are not intended to reproduce the full paper table by themselves; they are a stability and pipeline check.
- Full model output files and the private full dataset remain excluded for review-stage risk control.
- Gemini CoT metrics may have smaller valid N where the available CoT outputs are incomplete.
