# Synthetic neonatal IMU fixtures

These files are deterministic engineering test fixtures. They are **not infant recordings**, do not establish clinical accuracy, and must not be mixed with real observations without a `synthetic` provenance flag.

## What is generated

- `100hz/`: the modeled source signals. Use these for frequency-domain development.
- `10hz_legacy/`: every tenth source sample, matching the app's current serial output rate. These deliberately demonstrate aliasing and are not a faithful representation of fine neonatal tremor.
- `manifest.json`: seed, assumptions, expected class, event intervals, and measured summary values.

Each scenario is 30 seconds and contains synchronized accelerometer (`m/s²`), gyroscope (`rad/s`), and derived movement channels for two ICM-20948 sensors.

Scenarios include quiet/still, low-amplitude fine tremor, intermittent tremor with genuine still intervals, irregular spontaneous movement, caregiver handling, bed impacts, sensor slip, cable vibration, and mixed artifacts.

## Research constraints used

- A clinical review describes fine neonatal tremor as rhythmic, greater than 6 Hz, and lower amplitude than coarse tremor. It also warns that tremor/jitteriness is not equivalent to seizure or a diagnosis: <https://pmc.ncbi.nlm.nih.gov/articles/PMC2606074/>
- Infant wearable studies commonly use synchronized bilateral sensors and report rates of 20–30 Hz; caregiver handling, stroller/car motion, and similar common-mode activity are documented confounders: <https://pmc.ncbi.nlm.nih.gov/articles/PMC6651298/> and <https://pmc.ncbi.nlm.nih.gov/articles/PMC5558826/>
- A recent infant leg-movement study uses 25 Hz, ±16 g and ±2000 dps, and rejects implausibly dense mechanical movement sequences as artifact: <https://pmc.ncbi.nlm.nih.gov/articles/PMC13284690/>
- The ICM-20948 datasheet gives typical noise densities of 230 µg/√Hz for acceleration and 0.015 dps/√Hz for angular rate, plus configurable filtering and output rates: <https://product.tdk.com/system/files/dam/doc/product/sensor/mortion-inertial/imu/data_sheet/ds-000189-icm-20948-v1.5.pdf>

## Explicit assumptions

Published neonatal IMU amplitude distributions matching this exact board, attachment, placement, and population were not found. The fine-tremor displacement range (0.25–0.85 mm) is therefore a configurable engineering stress-test assumption—not a clinical claim. Each tremor uses frequency drift, amplitude modulation, independent sensor phase/axis, and smooth episode edges rather than a perfect sine wave.

The still scenario includes raw sensor noise and drift. Its derived `mov` channel is expected to remain below the current 0.12 m/s² activity threshold; clamping can make much of that derived channel zero even though raw axes remain noisy.

## Important sampling limitation

The Teensy currently samples at 100 Hz but prints only the latest reading at 10 Hz. A 10 Hz stream has a 5 Hz Nyquist limit, so a clinically described fine tremor above 6 Hz aliases into a false lower frequency. The 100 Hz source files are the appropriate fixtures for tremor work. The 10 Hz files are retained only to test the current legacy pipeline and quantify failure modes.

Regenerate with:

```powershell
python analysis\generate_synthetic_neonatal_imu.py --output synthetic-data
```
