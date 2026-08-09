# Four-IMU synthetic movement benchmark

This folder contains **synthetic engineering data, not infant recordings and
not clinical ground truth**.

- `train_generator_a.npz`: 144 nine-second sequences produced by analytic,
  gated movement primitives.
- `evaluation_generator_b.npz`: 72 sequences produced by a separate generator
  using chirps, colored noise, minimum-jerk bouts, and different artifact
  construction.
- Each row is time plus accelerometer XYZ and gyroscope XYZ for physical IMUs
  1–4 (25 columns total). Rates are 52, 60, and 100 Hz.
- JSON manifests contain every sequence ID, label, seed, sample rate, and the
  explicit `clinical_ground_truth: false` marker.
- `examples/` contains one readable four-IMU CSV per class from each generator.

The simulated sensor limits use the installed ICM-20948 configuration envelope
(±16 g accelerometer and ±2000 degrees/second gyroscope). Movement families are
informed by infant wearable protocols, but amplitudes and labels remain
engineering assumptions until checked against synchronized, clinician-reviewed
infant recordings.

Rebuild everything with:

```powershell
python analysis/sensor_ml_pipeline.py
npm run test:sensor-model
```
