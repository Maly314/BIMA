"""Leakage-resistant synthetic four-IMU movement model pipeline.

The outputs are engineering test data, not patient data or clinical ground
truth.  Generator A is used only for fitting; structurally different Generator
B is used only for final evaluation so overlapping synthetic waveforms cannot
leak into the reported test score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

G = 9.80665
SENSOR_COUNT = 4
AXES_PER_SENSOR = 6
LABELS = ("still_or_nonperiodic", "fine_tremor", "artifact_or_unusable")
SAMPLE_RATES = (52, 60, 100)
WINDOW_SECONDS = 3.0


@dataclass(frozen=True)
class SequenceSpec:
    family: str
    sequence_id: str
    label: str
    seed: int
    sample_rate_hz: int
    duration_s: float
    subtype: str


def _unit(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    return vector / max(np.linalg.norm(vector), 1e-12)


def _smooth_gate(t: np.ndarray, start: float, stop: float, edge: float) -> np.ndarray:
    return 0.5 * (np.tanh((t - start) / edge) - np.tanh((t - stop) / edge))


def _colored_noise(rng: np.random.Generator, shape: tuple[int, ...], alpha: float) -> np.ndarray:
    white = rng.normal(size=shape)
    out = np.empty_like(white)
    out[0] = white[0]
    scale = math.sqrt(max(1e-9, 1.0 - alpha * alpha))
    for index in range(1, len(out)):
        out[index] = alpha * out[index - 1] + scale * white[index]
    return out


def _base_signal(spec: SequenceSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec.seed)
    fs = spec.sample_rate_hz
    t = np.arange(round(spec.duration_s * fs), dtype=float) / fs
    accel = np.zeros((len(t), SENSOR_COUNT, 3), dtype=np.float32)
    gyro = np.zeros_like(accel)

    # Independent attachment orientations make magnitude and orientation-robust
    # features necessary. Noise and bias are randomized per physical unit.
    for sensor in range(SENSOR_COUNT):
        accel[:, sensor, :] = G * _unit(rng)
    accel += rng.normal(0, rng.uniform(0.008, 0.025), accel.shape)
    gyro += rng.normal(0, rng.uniform(0.0015, 0.006), gyro.shape)
    accel += np.cumsum(rng.normal(0, 2e-5, accel.shape), axis=0)
    gyro += rng.normal(0, 0.008, (1, SENSOR_COUNT, 3))
    if spec.family == "eval_b":
        accel += 0.012 * _colored_noise(rng, accel.shape, 0.88)
        gyro += 0.003 * _colored_noise(rng, gyro.shape, 0.72)
    return t, accel, gyro


def _add_train_nonperiodic(t: np.ndarray, accel: np.ndarray, gyro: np.ndarray, rng: np.random.Generator) -> str:
    if rng.random() < 0.35:
        return "quiet_stillness"
    for sensor in range(SENSOR_COUNT):
        for _ in range(rng.integers(2, 6)):
            start = rng.uniform(0.4, max(0.5, t[-1] - 1.0))
            duration = rng.uniform(0.35, 1.5)
            gate = _smooth_gate(t, start, start + duration, 0.12)
            frequency = rng.uniform(0.35, 2.3)
            phase = rng.uniform(0, 2 * np.pi)
            direction = _unit(rng)
            accel[:, sensor] += (rng.uniform(0.08, 1.2) * np.sin(2*np.pi*frequency*t + phase) * gate)[:, None] * direction
            gyro[:, sensor] += (rng.uniform(0.04, 0.7) * np.cos(2*np.pi*frequency*t + phase) * gate)[:, None] * np.roll(direction, 1)
    return "smooth_spontaneous_bouts"


def _add_eval_nonperiodic(t: np.ndarray, accel: np.ndarray, gyro: np.ndarray, rng: np.random.Generator) -> str:
    if rng.random() < 0.30:
        # Evaluation stillness has incubator sway absent from the training family.
        common = rng.uniform(0.006, 0.025) * np.sin(2*np.pi*rng.uniform(0.15, 0.42)*t + rng.uniform(0, 6))
        accel[:, :, rng.integers(0, 3)] += common[:, None]
        return "stillness_with_bed_sway"
    # Minimum-jerk pulses, not gated sinusoids used by Generator A.
    for sensor in range(SENSOR_COUNT):
        for _ in range(rng.integers(3, 7)):
            start = rng.uniform(0.3, max(0.4, t[-1] - 0.8))
            duration = rng.uniform(0.3, 1.4)
            u = np.clip((t - start) / duration, 0, 1)
            position = 10*u**3 - 15*u**4 + 6*u**5
            velocity = np.gradient(position, 1 / (len(t) / max(t[-1], 1e-6)))
            pulse = velocity * ((t >= start) & (t <= start + duration))
            direction = _unit(rng)
            accel[:, sensor] += (rng.uniform(0.04, 0.35) * pulse)[:, None] * direction
            gyro[:, sensor] += (rng.uniform(0.02, 0.16) * np.gradient(pulse))[:, None] * np.roll(direction, 1)
    return "minimum_jerk_spontaneous_bouts"


def _add_tremor(t: np.ndarray, accel: np.ndarray, gyro: np.ndarray, rng: np.random.Generator, family: str) -> str:
    fs = 1 / np.median(np.diff(t))
    affected = rng.choice(SENSOR_COUNT, size=rng.integers(1, SENSOR_COUNT + 1), replace=False)
    for sensor in affected:
        if family == "train_a":
            center = rng.uniform(6.3, 9.2)
            frequency = center + rng.uniform(0.08, 0.35) * np.sin(2*np.pi*rng.uniform(0.03, 0.12)*t)
            envelope = _smooth_gate(t, rng.uniform(0.1, 1.0), t[-1] - rng.uniform(0.1, 1.0), 0.18)
            subtype = "narrowband_sinusoidal"
        else:
            # Independent test generator: chirp + stochastic phase + a weak
            # second harmonic and intermittent raised-cosine bouts.
            start_hz, stop_hz = rng.uniform(5.8, 7.3), rng.uniform(8.0, 10.3)
            frequency = start_hz + (stop_hz - start_hz) * (t / max(t[-1], 1e-9))
            frequency += 0.10 * _colored_noise(rng, (len(t),), 0.95)
            envelope = np.zeros_like(t)
            for _ in range(rng.integers(1, 4)):
                start = rng.uniform(0, max(0.1, t[-1] - 1.5))
                envelope += _smooth_gate(t, start, min(t[-1], start + rng.uniform(1.0, 3.4)), 0.10)
            envelope = np.clip(envelope, 0, 1)
            subtype = "chirped_interharmonic"
        phase = 2*np.pi*np.cumsum(frequency) / fs + rng.uniform(0, 2*np.pi)
        displacement = rng.uniform(0.00012, 0.00075)
        angular = np.deg2rad(rng.uniform(0.12, 0.9))
        wave = np.sin(phase)
        if family == "eval_b":
            wave = 0.88 * wave + 0.12 * np.sin(2 * phase + rng.uniform(0, 6))
        direction = _unit(rng)
        accel[:, sensor] += (displacement * (2*np.pi*frequency)**2 * wave * envelope)[:, None] * direction
        gyro[:, sensor] += (angular * 2*np.pi*frequency * np.cos(phase) * envelope)[:, None] * np.roll(direction, 1)
    return subtype


def _add_artifact(t: np.ndarray, accel: np.ndarray, gyro: np.ndarray, rng: np.random.Generator, family: str) -> str:
    choice = rng.choice(("handling", "impact", "slip", "cable", "sensor_failure"))
    sensor = int(rng.integers(0, SENSOR_COUNT))
    if choice == "handling":
        gate = _smooth_gate(t, rng.uniform(0.3, 1.5), t[-1] - rng.uniform(0.2, 1.0), 0.2)
        if family == "train_a":
            motion = rng.uniform(0.8, 2.2) * np.sin(2*np.pi*rng.uniform(0.25, 1.2)*t) * gate
        else:
            motion = rng.uniform(0.5, 1.5) * _colored_noise(rng, (len(t),), 0.96) * gate
        accel[:, :, 0] += motion[:, None]
        gyro[:, :, 1] += (0.35 * np.gradient(motion))[:, None]
    elif choice == "impact":
        for at in rng.uniform(0.5, t[-1] - 0.3, rng.integers(1, 5)):
            elapsed = np.maximum(0, t - at)
            frequency = rng.uniform(13, 24) if family == "train_a" else rng.uniform(10, 29)
            ring = (t >= at) * np.exp(-elapsed / rng.uniform(0.06, 0.25)) * np.sin(2*np.pi*frequency*elapsed)
            direction = _unit(rng)
            accel += (rng.uniform(2.5, 10) * ring)[:, None, None] * direction[None, None, :]
            gyro += (rng.uniform(0.3, 1.6) * ring)[:, None, None] * np.roll(direction, 1)[None, None, :]
    elif choice == "slip":
        start = rng.uniform(0.5, t[-1] - 0.5)
        step = _smooth_gate(t, start, t[-1] + 1, 0.02 if family == "train_a" else 0.10)
        accel[:, sensor] += (rng.uniform(0.5, 2.0) * step)[:, None] * _unit(rng)
        gyro[:, sensor] += (rng.uniform(0.2, 1.0) * np.gradient(step))[:, None] * _unit(rng)
    elif choice == "cable":
        frequency = rng.uniform(17, 30)
        start = rng.uniform(0.2, t[-1] - 1.0)
        gate = _smooth_gate(t, start, min(t[-1], start + rng.uniform(0.5, 2.5)), 0.03)
        accel[:, sensor] += (rng.uniform(0.5, 2.0) * np.sin(2*np.pi*frequency*t) * gate)[:, None] * _unit(rng)
    else:
        start = int(rng.uniform(0.15, 0.65) * len(t))
        if family == "train_a":
            accel[start:, sensor] = accel[start, sensor]
            gyro[start:, sensor] = gyro[start, sensor]
        else:
            accel[start:, sensor] = np.nan
            gyro[start:, sensor] = np.nan
    return choice


def simulate_sequence(spec: SequenceSpec) -> np.ndarray:
    if spec.family not in {"train_a", "eval_b"}:
        raise ValueError("family must be train_a or eval_b")
    if spec.label not in LABELS:
        raise ValueError(f"unknown label {spec.label}")
    t, accel, gyro = _base_signal(spec)
    rng = np.random.default_rng(spec.seed + 31_337)
    if spec.label == LABELS[0]:
        _add_train_nonperiodic(t, accel, gyro, rng) if spec.family == "train_a" else _add_eval_nonperiodic(t, accel, gyro, rng)
    elif spec.label == LABELS[1]:
        _add_tremor(t, accel, gyro, rng, spec.family)
    else:
        _add_artifact(t, accel, gyro, rng, spec.family)
    values = np.concatenate((accel, gyro), axis=2)
    return np.concatenate((t[:, None], values.reshape(len(t), -1)), axis=1).astype(np.float32)


def make_specs(family: str, per_class: int, duration_s: float, seed: int) -> list[SequenceSpec]:
    specs: list[SequenceSpec] = []
    for label_index, label in enumerate(LABELS):
        for index in range(per_class):
            item_seed = seed + label_index * 100_000 + index * 997
            subtype = "independently_sampled"
            specs.append(SequenceSpec(family, f"{family}-{label_index}-{index:04d}", label, item_seed, SAMPLE_RATES[index % len(SAMPLE_RATES)], duration_s, subtype))
    return specs


def save_dataset(path: Path, specs: list[SequenceSpec]) -> None:
    arrays = {spec.sequence_id: simulate_sequence(spec) for spec in specs}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".manifest.json").write_text(json.dumps({
        "synthetic": True,
        "clinical_ground_truth": False,
        "generator_family": specs[0].family,
        "channels": ["t"] + [f"s{s}_{axis}" for s in range(1, 5) for axis in ("ax", "ay", "az", "gx", "gy", "gz")],
        "sequences": [asdict(spec) for spec in specs],
    }, indent=2), encoding="utf-8")


def write_sequence_csv(path: Path, sequence: np.ndarray) -> None:
    fields = ["t"] + [f"s{s}_{axis}" for s in range(1, 5) for axis in ("ax", "ay", "az", "gx", "gy", "gz")]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(fields)
        writer.writerows(sequence)


def _spectral_features(values: np.ndarray, fs: float) -> list[float]:
    values = values - np.mean(values)
    power = np.abs(np.fft.rfft(values * np.hanning(len(values)))) ** 2
    freq = np.fft.rfftfreq(len(values), 1 / fs)
    total = float(power[(freq >= 0.2) & (freq <= min(30, fs / 2))].sum()) + 1e-12
    ratios = [float(power[(freq >= lo) & (freq < hi)].sum() / total) for lo, hi in ((0.2, 3), (3, 6), (6, 10), (10, min(20, fs/2 + 1e-6)))]
    valid = (freq >= 0.2) & (freq <= min(20, fs/2))
    peak = float(freq[valid][np.argmax(power[valid])]) if np.any(valid) else 0.0
    normalized = power[valid] / max(float(power[valid].sum()), 1e-12)
    entropy = float(-np.sum(normalized * np.log(normalized + 1e-12)) / np.log(max(2, len(normalized))))
    concentration = float(np.max(normalized)) if len(normalized) else 0.0
    return ratios + [peak, entropy, concentration]


def extract_features(sequence: np.ndarray, window_seconds: float = WINDOW_SECONDS) -> tuple[np.ndarray, np.ndarray]:
    t = sequence[:, 0].astype(float)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    fs = 1 / float(np.median(np.diff(t)))
    if fs < 25:
        raise ValueError(f"sample rate {fs:.1f} Hz is below the 25 Hz minimum and can alias fine tremor")
    if fs > 250:
        raise ValueError(f"sample rate {fs:.1f} Hz is outside the validated range")
    width = max(16, round(window_seconds * fs))
    stride = width
    output: list[list[float]] = []
    starts: list[float] = []
    for start in range(0, len(sequence) - width + 1, stride):
        raw = sequence[start:start+width, 1:].reshape(width, SENSOR_COUNT, AXES_PER_SENSOR).astype(float)
        features: list[float] = [fs]
        accel_series: list[np.ndarray] = []
        gyro_series: list[np.ndarray] = []
        for sensor in range(SENSOR_COUNT):
            sensor_raw = raw[:, sensor]
            missing = float(np.mean(~np.isfinite(sensor_raw)))
            clean = sensor_raw.copy()
            for axis in range(AXES_PER_SENSOR):
                column = clean[:, axis]
                finite = np.isfinite(column)
                fill = float(np.median(column[finite])) if np.any(finite) else 0.0
                column[~finite] = fill
            accel_mag = np.linalg.norm(clean[:, :3], axis=1)
            accel_dyn = accel_mag - np.median(accel_mag)
            gyro_vec = clean[:, 3:] - np.median(clean[:, 3:], axis=0)
            gyro_mag = np.linalg.norm(gyro_vec, axis=1)
            accel_series.append(accel_dyn)
            gyro_series.append(gyro_mag)
            frozen = float(np.mean(np.std(clean, axis=0) < 1e-7))
            features.extend((missing, frozen))
            for values in (accel_dyn, gyro_mag):
                rms = float(np.sqrt(np.mean(values**2)))
                std = float(np.std(values))
                q95 = float(np.quantile(np.abs(values), 0.95))
                peak = float(np.max(np.abs(values)))
                centered = values - np.mean(values)
                kurtosis = float(np.mean(centered**4) / max(np.mean(centered**2)**2, 1e-12))
                features.extend((rms, std, q95, peak, kurtosis, *_spectral_features(values, fs)))
        # Pairwise evidence distinguishes common-mode handling from limb-specific
        # periodic motion and makes sensor swaps irrelevant.
        for series_group in (accel_series, gyro_series):
            correlations = []
            for left in range(SENSOR_COUNT):
                for right in range(left + 1, SENSOR_COUNT):
                    a, b = series_group[left], series_group[right]
                    correlations.append(float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-9 and np.std(b) > 1e-9 else 0.0)
            features.extend((float(np.mean(correlations)), float(np.max(correlations)), float(np.min(correlations))))
        output.append(features)
        starts.append(float(t[start]))
    if not output:
        raise ValueError("sequence is shorter than one model window")
    return np.asarray(output, dtype=np.float32), np.asarray(starts, dtype=np.float32)


def assess_data_quality(sequence: np.ndarray) -> dict:
    """Report per-IMU availability independently from the movement label."""
    raw = sequence[:, 1:].reshape(len(sequence), SENSOR_COUNT, AXES_PER_SENSOR).astype(float)
    sensors = []
    unavailable = 0
    for sensor in range(SENSOR_COUNT):
        values = raw[:, sensor]
        missing_fraction = float(np.mean(~np.isfinite(values)))
        axis_std = np.asarray([
            np.std(values[np.isfinite(values[:, axis]), axis]) if np.any(np.isfinite(values[:, axis])) else 0.0
            for axis in range(AXES_PER_SENSOR)
        ])
        frozen_fraction = float(np.mean(np.nan_to_num(axis_std) < 1e-7))
        state = "unavailable" if missing_fraction >= 0.5 or frozen_fraction >= 0.8 else "degraded" if missing_fraction > 0.02 or frozen_fraction > 0 else "ok"
        unavailable += state == "unavailable"
        sensors.append({"imu": sensor + 1, "state": state, "missing_fraction": missing_fraction, "frozen_axis_fraction": frozen_fraction})
    overall = "unusable" if unavailable >= 2 else "degraded" if unavailable or any(item["state"] == "degraded" for item in sensors) else "ok"
    return {"overall": overall, "sensors": sensors}


def score_sequence(bundle: dict, sequence: np.ndarray) -> dict:
    features, starts = extract_features(sequence, bundle.get("window_seconds", WINDOW_SECONDS))
    probabilities = bundle["model"].predict_proba(features)
    predictions = bundle["model"].predict(features)
    return {
        "quality": assess_data_quality(sequence),
        "windows": [
            {"start_s": float(start), "prediction": bundle["labels"][int(prediction)], "probabilities": {label: float(probability) for label, probability in zip(bundle["labels"], row)}}
            for start, prediction, row in zip(starts, predictions, probabilities)
        ],
    }


def read_capture_csv(path: Path) -> np.ndarray:
    """Load BIMA CSVs while retaining absent physical IMUs as NaN columns."""
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows or "t" not in rows[0]:
        raise ValueError("capture CSV is empty or has no t column")
    output = np.full((len(rows), 1 + SENSOR_COUNT * AXES_PER_SENSOR), np.nan, dtype=np.float32)
    for row_index, row in enumerate(rows):
        try:
            output[row_index, 0] = float(row["t"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid timestamp at CSV row {row_index + 2}") from error
        for sensor in range(1, SENSOR_COUNT + 1):
            columns = [f"s{sensor}_{axis}" for axis in ("ax", "ay", "az", "gx", "gy", "gz")]
            present = [column in row for column in columns]
            if any(present) and not all(present):
                raise ValueError(f"IMU {sensor} has only a partial set of raw axes")
            if all(present):
                for axis, column in enumerate(columns):
                    value = row[column]
                    if value not in (None, ""):
                        try:
                            output[row_index, 1 + (sensor - 1) * AXES_PER_SENSOR + axis] = float(value)
                        except ValueError as error:
                            raise ValueError(f"non-numeric {column} at CSV row {row_index + 2}") from error
    return output


def _load_dataset(path: Path) -> tuple[list[SequenceSpec], dict[str, np.ndarray]]:
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    specs = [SequenceSpec(**item) for item in manifest["sequences"]]
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return specs, arrays


def _feature_matrix(specs: Iterable[SequenceSpec], arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, groups = [], [], []
    for spec in specs:
        features, _ = extract_features(arrays[spec.sequence_id])
        xs.append(features)
        ys.extend([LABELS.index(spec.label)] * len(features))
        groups.extend([spec.sequence_id] * len(features))
    return np.vstack(xs), np.asarray(ys), np.asarray(groups)


def train_and_evaluate(train_path: Path, evaluation_path: Path, model_path: Path, report_path: Path) -> dict:
    train_specs, train_arrays = _load_dataset(train_path)
    eval_specs, eval_arrays = _load_dataset(evaluation_path)
    if {item.family for item in train_specs} != {"train_a"} or {item.family for item in eval_specs} != {"eval_b"}:
        raise ValueError("training must use train_a and final evaluation must use independent eval_b")
    if {item.seed for item in train_specs} & {item.seed for item in eval_specs}:
        raise ValueError("training and evaluation seeds overlap")
    train_x, train_y, train_groups = _feature_matrix(train_specs, train_arrays)
    eval_x, eval_y, eval_groups = _feature_matrix(eval_specs, eval_arrays)
    model = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=0.8, random_state=73)
    model.fit(train_x, train_y)
    predictions = model.predict(eval_x)
    probabilities = model.predict_proba(eval_x)
    evaluation_rates = np.asarray([{spec.sequence_id: spec.sample_rate_hz for spec in eval_specs}[group] for group in eval_groups])
    report = {
        "purpose": "engineering classification of synthetic movement windows; not diagnosis",
        "clinical_ground_truth": False,
        "classes": list(LABELS),
        "window_seconds": WINDOW_SECONDS,
        "training_generator": "train_a",
        "evaluation_generator": "eval_b (independent signal construction)",
        "training_sequences": len(set(train_groups)),
        "evaluation_sequences": len(set(eval_groups)),
        "training_windows": len(train_y),
        "evaluation_windows": len(eval_y),
        "confusion_matrix": confusion_matrix(eval_y, predictions, labels=range(len(LABELS))).tolist(),
        "classification_report": classification_report(eval_y, predictions, target_names=LABELS, output_dict=True, zero_division=0),
        "mean_confidence": float(np.mean(np.max(probabilities, axis=1))),
        "per_sample_rate_accuracy": {
            str(rate): float(np.mean(predictions[evaluation_rates == rate] == eval_y[evaluation_rates == rate]))
            for rate in SAMPLE_RATES
        },
        "important_limit": "Synthetic performance cannot establish performance on infants, cerebral palsy, a new attachment, or real clinical artifacts.",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": LABELS, "window_seconds": WINDOW_SECONDS, "feature_count": train_x.shape[1], "version": 1}, model_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("synthetic-data/v2"))
    parser.add_argument("--model", type=Path, default=Path("local-models/four-imu-synthetic.joblib"))
    parser.add_argument("--report", type=Path, default=Path("local-models/four-imu-synthetic-report.json"))
    parser.add_argument("--train-per-class", type=int, default=48)
    parser.add_argument("--eval-per-class", type=int, default=24)
    parser.add_argument("--duration", type=float, default=9.0)
    parser.add_argument("--predict", type=Path, help="score a captured 25-250 Hz BIMA sensor CSV with the existing model")
    args = parser.parse_args()
    if args.predict:
        print(json.dumps(score_sequence(joblib.load(args.model), read_capture_csv(args.predict)), indent=2))
        return
    train_path = args.output / "train_generator_a.npz"
    eval_path = args.output / "evaluation_generator_b.npz"
    save_dataset(train_path, make_specs("train_a", args.train_per_class, args.duration, 20_948))
    evaluation_specs = make_specs("eval_b", args.eval_per_class, args.duration, 920_948)
    save_dataset(eval_path, evaluation_specs)
    # Human-readable examples make the archive inspectable without loading NPZ.
    for family, specs in (("train_a", make_specs("train_a", 1, args.duration, 20_948)), ("eval_b", make_specs("eval_b", 1, args.duration, 920_948))):
        for spec in specs:
            write_sequence_csv(args.output / "examples" / f"{family}-{spec.label}.csv", simulate_sequence(spec))
    print(json.dumps(train_and_evaluate(train_path, eval_path, args.model, args.report), indent=2))


if __name__ == "__main__":
    main()
