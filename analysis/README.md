# Exploratory movement baseline

This local model classifies overlapping five-second windows as `sustained_shaky`
or `not_shaky_or_artifact`. It is for adult test data only and is not a medical
model. Raw accelerometer and gyroscope channels are normalized consistently so
recordings made before and after the calibration update can be compared.

Train with the two labeled recordings:

```powershell
python analysis/train_movement_model.py `
  --shaky C:\Users\mwstr\Downloads\movement-sensors-1784658505915.csv `
  --still C:\Users\mwstr\Downloads\movement-sensors-1784658593112.csv `
  --artifact C:\Users\mwstr\Downloads\patient-3-susp-wk1-sensors.csv
```

Score a new recording:

```powershell
python analysis/train_movement_model.py `
  --model local-models/movement-baseline.joblib `
  --predict C:\path\to\new-recording.csv
```

The artifact recording is a hard negative: brief impulses should not become
"sustained shaky." Each recording is balanced so the longer artifact file does
not dominate training. The report's cross-validation is still only a
within-recording smoke test. A future model needs multiple independent sessions,
people, placements, sensor sets, and artifact recordings before it can be
evaluated honestly.

The data-quality companion is `analysis/artifact_data_quality.ipynb`.
