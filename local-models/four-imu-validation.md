# Four-IMU synthetic model validation

Validated 2026-08-09 with Generator A used only for training and the separately
implemented Generator B used only for evaluation.

## Model result

- 144 training sequences / 432 non-overlapping 3-second windows
- 72 independent evaluation sequences / 216 windows
- Overall accuracy: 68.98%
- Macro F1: 70.04%
- Fine-tremor precision: 100.00%
- Fine-tremor recall: 76.39%
- Artifact/unusable recall: 62.50%
- Accuracy by rate: 52 Hz 69.44%, 60 Hz 68.06%, 100 Hz 69.44%

The modest result is intentionally retained rather than tuning against the
evaluation generator. It documents that unfamiliar movement/artifact patterns
remain a real limitation.

## Automated checks

The sensor/model suite covers all four IMUs and 24 raw axes, deterministic
generation, train/evaluation separation, seed leakage, axis rotation, sensor
reordering, packet loss, timing jitter, missing/frozen sensors, two-sensor
failure, invalid timestamps, low-rate aliasing, ICM-20948 full-scale bounds,
BIMA CSV schema parsing, serialization, rate-specific performance, and fixed
performance floors. It runs as part of `npm test`.

## Use

```powershell
python analysis/sensor_ml_pipeline.py --predict path\to\100hz-capture.csv
```

This is an engineering classifier for `still_or_nonperiodic`, `fine_tremor`,
and `artifact_or_unusable`. It is not a cerebral-palsy detector and cannot be
used for clinical decisions without subject-held-out real infant validation,
protocol review, and the required clinical/regulatory controls.
