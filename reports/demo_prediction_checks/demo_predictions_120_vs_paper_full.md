# Demo Predictions 120 vs Paper Full Metrics

Values are `paper full / demo120 / delta`, where delta is `demo120 - paper full`.

| Setting | S1L | S2L | S3L | TSMR | CSGR | CRR | ML-MF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.3870 / 0.3750 / -0.0120 | 0.4040 / 0.3917 / -0.0123 | 0.4750 / 0.4750 / +0.0000 | 0.4220 / 0.4139 / -0.0081 | 0.4090 / 0.4110 / +0.0020 | 0.4270 / 0.4255 / -0.0015 | 0.3490 / 0.3394 / -0.0096 |
| CoT | 0.4200 / 0.4000 / -0.0200 | 0.4280 / 0.4333 / +0.0053 | 0.4960 / 0.5167 / +0.0207 | 0.4480 / 0.4500 / +0.0020 | 0.3280 / 0.3529 / +0.0249 | 0.2800 / 0.2692 / -0.0108 | 0.3720 / 0.3731 / +0.0011 |
| ArtTIDE | 0.4730 / 0.4833 / +0.0103 | 0.5090 / 0.4833 / -0.0257 | 0.5660 / 0.5500 / -0.0160 | 0.5160 / 0.5056 / -0.0104 | 0.4370 / 0.4194 / -0.0176 | 0.3100 / 0.3103 / +0.0003 | 0.4260 / 0.4108 / -0.0152 |

The 120-item demo prediction set is intended for workflow validation and metric recomputation, not as the complete experimental output release.
