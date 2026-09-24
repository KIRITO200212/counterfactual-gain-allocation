# Numerical reproduction

All values are scene-equal. Intervals are paired percentile bootstrap with 10,000 resamples and seed 20270908. ReG training seeds remain separate.

| Endpoint | Seed A (1234) | Seed B (20260910) |
|---|---:|---:|
| ReG − continued, contrast gain (nats/token) | 0.04214227 [0.03609907, 0.04827601] | 0.02170957 [0.01742148, 0.02593048] |
| Control share | 0.85130015 [0.75758393, 0.96108483] | 0.89870119 [0.78374978, 1.05054920] |
| Control − edit contribution gap | 0.02960917 [0.02298349, 0.03599611] | 0.01731126 [0.01300517, 0.02154878] |

| Bridge: Full − Prefix-only | Estimate | 95% CI |
|---|---:|---:|
| boundary_mapped_probability (percentage points) | 1.55105291 | [0.59598941, 2.48488198] |
| suffix_mapped_rate (percentage points) | 1.48809524 | [0.29761905, 2.82738095] |
| full_mapped_rate (percentage points) | 3.57142857 | [-2.38095238, 9.52380952] |

Run python scripts/reproduce_statistics.py for full-precision checks of the exported observations. The checked original analysis uses fitted-model differences, not within-training trajectories. These intervals are conditional on checkpoints.
